#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(_WIN32)
#define MINIAI_API extern "C" __declspec(dllexport)
#else
#define MINIAI_API extern "C" __attribute__((visibility("default")))
#endif
namespace { typedef std::vector<std::pair<int, double>> Sparse; struct Layer { int in = 0, out = 0; std::vector<double> W;
std::vector<double> b; }; struct Bot { std::vector<std::string> inputs; std::vector<int> labels; std::vector<std::string> responses;
std::unordered_map<std::string, int> respIndex; std::unordered_map<std::string, int> vocab; std::vector<std::string> vocabList;
std::vector<double> idf; int mode = 0; std::vector<double> linW; std::vector<Layer> layers; double threshold = -1.0; std::string fallback =
"Sorry, I don't understand that yet."; double lastConf = 0.0; std::string lastErr; bool verbose = false; unsigned seed = 42; };
std::vector<std::string> features(const std::string& text) { std::vector<std::string> toks; std::string cur; for (unsigned char c : text) {
if (c >= 128 || std::isalnum(c)) { cur.push_back(c >= 128 ? (char)c : (char)std::tolower(c)); } else if (!cur.empty()) {
toks.push_back(cur); cur.clear(); } } if (!cur.empty()) toks.push_back(cur); std::vector<std::string> f = toks; for (size_t i = 0; i + 1 <
toks.size(); ++i) f.push_back(toks[i] + "_" + toks[i + 1]); return f; } void buildVocab(Bot& b) { b.vocab.clear(); b.vocabList.clear();
b.idf.clear(); std::vector<int> df; for (const auto& s : b.inputs) { std::vector<std::string> f = features(s); std::sort(f.begin(),
f.end()); f.erase(std::unique(f.begin(), f.end()), f.end()); for (const auto& w : f) { auto it = b.vocab.find(w); if (it == b.vocab.end()) {
b.vocab[w] = (int)b.vocabList.size(); b.vocabList.push_back(w); df.push_back(1); } else df[it->second]++; } } double N =
(double)b.inputs.size(); for (int d : df) b.idf.push_back(std::log((N + 1.0) / (d + 1.0)) + 1.0); } Sparse vectorize(const Bot& b, const
std::string& text) { std::vector<std::string> f = features(text); std::sort(f.begin(), f.end()); f.erase(std::unique(f.begin(), f.end()),
f.end()); double unk = std::log((double)b.inputs.size() + 1.0) + 1.0; Sparse x; double sq = 0; for (const auto& w : f) { auto it =
b.vocab.find(w); if (it != b.vocab.end() && (size_t)it->second < b.idf.size()) { double v = b.idf[it->second];
x.push_back(std::make_pair(it->second, v)); sq += v * v; } else sq += unk * unk; } std::sort(x.begin(), x.end()); if (sq > 0) { double n =
std::sqrt(sq); for (auto& p : x) p.second /= n; } return x; } double sparseDot(const Sparse& a, const Sparse& b) { size_t i = 0, j = 0;
double s = 0; while (i < a.size() && j < b.size()) { if (a[i].first == b[j].first) s += a[i++].second * b[j++].second; else if (a[i].first <
b[j].first) ++i; else ++j; } return s; } bool cholSolve(std::vector<double>& A, int n, std::vector<double>& B, int m) { for (int i = 0; i <
n; ++i) { for (int j = 0; j <= i; ++j) { double s = A[(size_t)i * n + j]; for (int k = 0; k < j; ++k) s -= A[(size_t)i * n + k] *
A[(size_t)j * n + k]; if (i == j) { if (s <= 1e-12) return false; A[(size_t)i * n + i] = std::sqrt(s); } else { A[(size_t)i * n + j] = s /
A[(size_t)j * n + j]; } } } for (int c = 0; c < m; ++c) { for (int i = 0; i < n; ++i) { double s = B[(size_t)i * m + c]; for (int k = 0; k <
i; ++k) s -= A[(size_t)i * n + k] * B[(size_t)k * m + c]; B[(size_t)i * m + c] = s / A[(size_t)i * n + i]; } for (int i = n - 1; i >= 0;
--i) { double s = B[(size_t)i * m + c]; for (int k = i + 1; k < n; ++k) s -= A[(size_t)k * n + i] * B[(size_t)k * m + c]; B[(size_t)i * m +
c] = s / A[(size_t)i * n + i]; } } return true; } void softmax(std::vector<double>& v) { double mx = *std::max_element(v.begin(), v.end()),
s = 0; for (auto& x : v) { x = std::exp(x - mx); s += x; } for (auto& x : v) x /= s; } void forward(const std::vector<Layer>& L, const
Sparse& x, std::vector<std::vector<double>>& a) { a.resize(L.size() + 1); for (size_t l = 0; l < L.size(); ++l) { const Layer& ly = L[l];
std::vector<double>& o = a[l + 1]; o = ly.b; if (l == 0) { for (const auto& p : x) { const double* w = &ly.W[(size_t)p.first * ly.out]; for
(int j = 0; j < ly.out; ++j) o[j] += p.second * w[j]; } } else { const std::vector<double>& in = a[l]; for (int i = 0; i < ly.in; ++i) {
const double* w = &ly.W[(size_t)i * ly.out]; double v = in[i]; for (int j = 0; j < ly.out; ++j) o[j] += v * w[j]; } } if (l + 1 < L.size())
for (auto& v : o) v = std::tanh(v); else softmax(o); } } void adamStep(std::vector<double>& p, const std::vector<double>& g,
std::vector<double>& m, std::vector<double>& v, double lr, double scale, long t) { const double b1 = 0.9, b2 = 0.999; double c1 = 1.0 -
std::pow(b1, (double)t), c2 = 1.0 - std::pow(b2, (double)t); for (size_t i = 0; i < p.size(); ++i) { double gi = g[i] * scale; m[i] = b1 *
m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi; p[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + 1e-8); } } int predict(const
Bot& b, const std::string& text, double& conf) { conf = 0.0; if (b.mode == 0 || b.responses.empty()) return -1; Sparse x = vectorize(b,
text); if (x.empty()) return -1; int C = (int)b.responses.size(); std::vector<double> s(C, 0.0); if (b.mode == 1) { int F =
(int)b.vocabList.size(); for (const auto& p : x) for (int c = 0; c < C; ++c) s[c] += p.second * b.linW[(size_t)p.first * C + c]; for (int c
= 0; c < C; ++c) s[c] += b.linW[(size_t)F * C + c]; int best = (int)(std::max_element(s.begin(), s.end()) - s.begin()); conf = std::min(1.0,
std::max(0.0, s[best])); return best; } std::vector<std::vector<double>> a; forward(b.layers, x, a); const std::vector<double>& p =
a.back(); int best = (int)(std::max_element(p.begin(), p.end()) - p.begin()); conf = p[best]; return best; } double accuracy(const Bot& b) {
if (b.inputs.empty()) return 0.0; int ok = 0; for (size_t i = 0; i < b.inputs.size(); ++i) { double c; if (predict(b, b.inputs[i], c) ==
b.labels[i]) ++ok; } return (double)ok / (double)b.inputs.size(); } double fail(Bot& b, const std::string& msg) { b.lastErr = msg; return
-1.0; } double doLinear(Bot& b, double lambda) { int N = (int)b.inputs.size(), C = (int)b.responses.size(); if (N == 0) return fail(b,
"no training data: call ai_add_pair first"); if (lambda <= 0) lambda = 1e-3; buildVocab(b); int F = (int)b.vocabList.size(); if (F == 0)
return fail(b, "training inputs contain no usable words"); int D = F + 1; if (std::min(N, D) > 4000) return fail(b,
"dataset too large for trainLinearAlgebra (limit 4000); use trainComplex"); std::vector<Sparse> X(N); for (int i = 0; i < N; ++i) { X[i] =
vectorize(b, b.inputs[i]); X[i].push_back(std::make_pair(F, 1.0)); } std::vector<double> W((size_t)D * C, 0.0); if (D <= N) {
std::vector<double> A((size_t)D * D, 0.0), B((size_t)D * C, 0.0); for (int i = 0; i < N; ++i) for (const auto& p : X[i]) { for (const auto&
q : X[i]) A[(size_t)p.first * D + q.first] += p.second * q.second; B[(size_t)p.first * C + b.labels[i]] += p.second; } for (int d = 0; d <
D; ++d) A[(size_t)d * D + d] += lambda; if (!cholSolve(A, D, B, C)) return fail(b, "linear system was not solvable"); W = B; } else {
std::vector<double> A((size_t)N * N, 0.0), Y((size_t)N * C, 0.0); for (int i = 0; i < N; ++i) { for (int j = i; j < N; ++j) { double d =
sparseDot(X[i], X[j]); A[(size_t)i * N + j] = d; A[(size_t)j * N + i] = d; } A[(size_t)i * N + i] += lambda; Y[(size_t)i * C + b.labels[i]]
= 1.0; } if (!cholSolve(A, N, Y, C)) return fail(b, "linear system was not solvable"); for (int i = 0; i < N; ++i) for (const auto& p :
X[i]) for (int c = 0; c < C; ++c) W[(size_t)p.first * C + c] += p.second * Y[(size_t)i * C + c]; } b.linW = W; b.layers.clear(); b.mode = 1;
double acc = accuracy(b); if (b.verbose) std::fprintf(stderr, "[miniai] linear algebra solve done, train accuracy %.1f%%\n", acc * 100);
return acc; } double doComplex(Bot& b, int h1, int h2, int epochs, double lr) { int N = (int)b.inputs.size(), C = (int)b.responses.size();
if (N == 0) return fail(b, "no training data: call ai_add_pair first"); if (epochs <= 0) epochs = 300; if (lr <= 0) lr = 0.02;
buildVocab(b); int F = (int)b.vocabList.size(); if (F == 0) return fail(b, "training inputs contain no usable words"); std::vector<int>
sizes; sizes.push_back(F); if (h1 > 0) sizes.push_back(h1); if (h1 > 0 && h2 > 0) sizes.push_back(h2); sizes.push_back(C); size_t L =
sizes.size() - 1; std::mt19937 rng(b.seed); std::vector<Layer> layers(L); for (size_t l = 0; l < L; ++l) { Layer& ly = layers[l]; ly.in =
sizes[l]; ly.out = sizes[l + 1]; double lim = std::sqrt(6.0 / (ly.in + ly.out)); std::uniform_real_distribution<double> U(-lim, lim);
ly.W.resize((size_t)ly.in * ly.out); for (auto& w : ly.W) w = U(rng); ly.b.assign(ly.out, 0.0); } std::vector<Sparse> X(N); for (int i = 0;
i < N; ++i) X[i] = vectorize(b, b.inputs[i]); std::vector<std::vector<double>> mW(L), vW(L), mb(L), vb(L), gW(L), gb(L); for (size_t l = 0;
l < L; ++l) { mW[l].assign(layers[l].W.size(), 0.0); vW[l] = mW[l]; gW[l] = mW[l]; mb[l].assign(layers[l].b.size(), 0.0); vb[l] = mb[l];
gb[l] = mb[l]; } const int batch = std::min(N, 8); std::vector<int> order(N); std::iota(order.begin(), order.end(), 0);
std::vector<std::vector<double>> a; long step = 0; int every = std::max(1, epochs / 10); for (int ep = 1; ep <= epochs; ++ep) {
std::shuffle(order.begin(), order.end(), rng); double lossSum = 0; for (int start = 0; start < N; start += batch) { int end = std::min(N,
start + batch); for (size_t l = 0; l < L; ++l) { std::fill(gW[l].begin(), gW[l].end(), 0.0); std::fill(gb[l].begin(), gb[l].end(), 0.0); }
for (int k = start; k < end; ++k) { int i = order[k], y = b.labels[i]; forward(layers, X[i], a); std::vector<double> delta = a.back();
lossSum += -std::log(std::max(delta[y], 1e-12)); delta[y] -= 1.0; for (int l = (int)L - 1; l >= 0; --l) { const Layer& ly = layers[l]; for
(int j = 0; j < ly.out; ++j) gb[l][j] += delta[j]; if (l == 0) { for (const auto& p : X[i]) for (int j = 0; j < ly.out; ++j)
gW[0][(size_t)p.first * ly.out + j] += p.second * delta[j]; } else { const std::vector<double>& in = a[l]; for (int r = 0; r < ly.in; ++r)
for (int j = 0; j < ly.out; ++j) gW[l][(size_t)r * ly.out + j] += in[r] * delta[j]; std::vector<double> nd(ly.in, 0.0); for (int r = 0; r <
ly.in; ++r) { double s = 0; for (int j = 0; j < ly.out; ++j) s += ly.W[(size_t)r * ly.out + j] * delta[j]; nd[r] = s * (1.0 - in[r] *
in[r]); } delta.swap(nd); } } } ++step; double scale = 1.0 / (end - start); for (size_t l = 0; l < L; ++l) { adamStep(layers[l].W, gW[l],
mW[l], vW[l], lr, scale, step); adamStep(layers[l].b, gb[l], mb[l], vb[l], lr, scale, step); } } double loss = lossSum / N; if (b.verbose &&
(ep % every == 0 || ep == 1)) std::fprintf(stderr, "[miniai] epoch %d/%d  loss %.5f\n", ep, epochs, loss); if (loss < 1e-3) { if (b.verbose)
std::fprintf(stderr, "[miniai] converged at epoch %d\n", ep); break; } } b.layers = layers; b.linW.clear(); b.mode = 2; return accuracy(b);
} std::string esc(const std::string& s) { std::string o; for (char c : s) { if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else
if (c == '\r') o += "\\r"; else if (c == '\t') o += "\\t"; else o += c; } return o; } std::string unesc(const std::string& s) { std::string
o; for (size_t i = 0; i < s.size(); ++i) { if (s[i] == '\\' && i + 1 < s.size()) { char n = s[++i]; o += (n == 'n') ? '\n' : (n == 'r') ?
'\r' : (n == 't') ? '\t' : n; } else o += s[i]; } return o; } int doSave(Bot& b, const char* path) { std::ofstream o(path); if (!o) {
b.lastErr = std::string("cannot open file for writing: ") + path; return -1; } o << std::setprecision(17); o << "MINIAI 1\n"; o << "mode "
<< b.mode << "\n"; o << "threshold " << b.threshold << "\n"; o << "fallback " << esc(b.fallback) << "\n"; o << "responses " <<
b.responses.size() << "\n"; for (const auto& r : b.responses) o << esc(r) << "\n"; o << "examples " << b.inputs.size() << "\n"; for (size_t
i = 0; i < b.inputs.size(); ++i) o << b.labels[i] << "\t" << esc(b.inputs[i]) << "\n"; o << "vocab " << b.vocabList.size() << "\n"; for
(const auto& w : b.vocabList) o << w << "\n"; o << "idf " << b.idf.size() << "\n"; for (double v : b.idf) o << v << " "; o << "\n"; if
(b.mode == 1) { o << "linear " << b.linW.size() << "\n"; for (double v : b.linW) o << v << " "; o << "\n"; } else if (b.mode == 2) { o <<
"layers " << b.layers.size() << "\n"; for (const auto& ly : b.layers) { o << "layer " << ly.in << " " << ly.out << "\n"; for (double v :
ly.W) o << v << " "; o << "\n"; for (double v : ly.b) o << v << " "; o << "\n"; } } o << "END\n"; if (!o) { b.lastErr = "write error";
return -1; } return 0; } bool readKey(std::istream& in, const char* key, std::string& rest) { std::string line; if (!std::getline(in, line))
return false; if (!line.empty() && line.back() == '\r') line.pop_back(); size_t sp = line.find(' '); std::string k = (sp ==
std::string::npos) ? line : line.substr(0, sp); if (k != key) return false; rest = (sp == std::string::npos) ? "" : line.substr(sp + 1);
return true; } bool readDoubles(std::istream& in, size_t n, std::vector<double>& out) { std::string line; if (!std::getline(in, line))
return n == 0; std::istringstream ss(line); out.resize(n); for (size_t i = 0; i < n; ++i) if (!(ss >> out[i])) return false; return true; }
int doLoad(Bot& b, const char* path) { std::ifstream in(path); if (!in) { b.lastErr = std::string("cannot open file: ") + path; return -1; }
Bot t; std::string rest; auto bad = [&](const char* what) { b.lastErr = std::string("bad model file near: ") + what; return -1; }; try { if
(!readKey(in, "MINIAI", rest) || rest != "1") return bad("header"); if (!readKey(in, "mode", rest)) return bad("mode"); t.mode =
std::stoi(rest); if (!readKey(in, "threshold", rest)) return bad("threshold"); t.threshold = std::stod(rest); if (!readKey(in, "fallback",
rest)) return bad("fallback"); t.fallback = unesc(rest); if (!readKey(in, "responses", rest)) return bad("responses"); size_t R =
std::stoul(rest); std::string line; for (size_t i = 0; i < R; ++i) { if (!std::getline(in, line)) return bad("response list"); if
(!line.empty() && line.back() == '\r') line.pop_back(); std::string r = unesc(line); t.respIndex[r] = (int)t.responses.size();
t.responses.push_back(r); } if (!readKey(in, "examples", rest)) return bad("examples"); size_t N = std::stoul(rest); for (size_t i = 0; i <
N; ++i) { if (!std::getline(in, line)) return bad("example list"); if (!line.empty() && line.back() == '\r') line.pop_back(); size_t tab =
line.find('\t'); if (tab == std::string::npos) return bad("example line"); int lab = std::stoi(line.substr(0, tab)); if (lab < 0 ||
(size_t)lab >= R) return bad("example label"); t.labels.push_back(lab); t.inputs.push_back(unesc(line.substr(tab + 1))); } if (!readKey(in,
"vocab", rest)) return bad("vocab"); size_t F = std::stoul(rest); for (size_t i = 0; i < F; ++i) { if (!std::getline(in, line)) return
bad("vocab list"); if (!line.empty() && line.back() == '\r') line.pop_back(); t.vocab[line] = (int)t.vocabList.size();
t.vocabList.push_back(line); } if (!readKey(in, "idf", rest)) return bad("idf"); if (std::stoul(rest) != F) return bad("idf size"); if
(!readDoubles(in, F, t.idf)) return bad("idf values"); if (t.mode == 1) { if (!readKey(in, "linear", rest)) return bad("linear"); size_t n =
std::stoul(rest); if (n != (F + 1) * R) return bad("linear size"); if (!readDoubles(in, n, t.linW)) return bad("linear weights"); } else if
(t.mode == 2) { if (!readKey(in, "layers", rest)) return bad("layers"); size_t L = std::stoul(rest); if (L == 0) return bad("layer count");
t.layers.resize(L); for (size_t l = 0; l < L; ++l) { if (!readKey(in, "layer", rest)) return bad("layer"); std::istringstream ss(rest);
Layer& ly = t.layers[l]; if (!(ss >> ly.in >> ly.out) || ly.in <= 0 || ly.out <= 0) return bad("layer dims"); if (!readDoubles(in,
(size_t)ly.in * ly.out, ly.W)) return bad("layer weights"); if (!readDoubles(in, (size_t)ly.out, ly.b)) return bad("layer bias"); if (l > 0
&& t.layers[l - 1].out != ly.in) return bad("layer chain"); } if ((size_t)t.layers.front().in != F || (size_t)t.layers.back().out != R)
return bad("layer shape"); } else if (t.mode != 0) { return bad("mode value"); } if (!readKey(in, "END", rest)) return bad("END marker"); }
catch (...) { return bad("number format"); } t.verbose = b.verbose; t.seed = b.seed; b = std::move(t); return 0; } Bot* get(void* h) {
return static_cast<Bot*>(h); } } MINIAI_API void* ai_create() { try { return new Bot(); } catch (...) { return nullptr; } } MINIAI_API void
ai_destroy(void* h) { delete get(h); } MINIAI_API int ai_add_pair(void* h, const char* input, const char* response) { Bot* b = get(h); if
(!b || !input || !response) return -1; try { auto it = b->respIndex.find(response); int id; if (it == b->respIndex.end()) { id =
(int)b->responses.size(); b->responses.push_back(response); b->respIndex[response] = id; } else id = it->second; b->inputs.push_back(input);
b->labels.push_back(id); b->mode = 0; b->layers.clear(); b->linW.clear(); return (int)b->inputs.size(); } catch (...) { b->lastErr =
"out of memory"; return -1; } } MINIAI_API int LOADPAIRS(void* h, const char* path) { Bot* b = get(h); if (!b || !path) return -1;
std::ifstream in(path); if (!in) { b->lastErr = std::string("cannot open file: ") + path; return -1; } std::string line; int added = 0;
while (std::getline(in, line)) { if (!line.empty() && line.back() == '\r') line.pop_back(); if (line.empty() || line[0] == '#') continue;
size_t bar = line.find('|'); if (bar == std::string::npos || bar == 0) continue; if (ai_add_pair(h, line.substr(0, bar).c_str(),
line.substr(bar + 1).c_str()) < 0) return -1; ++added; } return added; } MINIAI_API void ai_clear(void* h) { Bot* b = get(h); if (!b)
return; bool v = b->verbose; unsigned s = b->seed; double th = b->threshold; std::string fb = b->fallback; *b = Bot(); b->verbose = v;
b->seed = s; b->threshold = th; b->fallback = fb; } MINIAI_API double trainLinearAlgebra(void* h, double lambda) { Bot* b = get(h); if (!b)
return -1.0; try { return doLinear(*b, lambda); } catch (...) { return fail(*b, "exception while training"); } } MINIAI_API double
trainComplex(void* h, int hidden1, int hidden2, int epochs, double lr) { Bot* b = get(h); if (!b) return -1.0; try { return doComplex(*b,
hidden1, hidden2, epochs, lr); } catch (...) { return fail(*b, "exception while training"); } } MINIAI_API int SAVETOTXT(void* h, const
char* file) { Bot* b = get(h); if (!b || !file) return -1; try { return doSave(*b, file); } catch (...) { b->lastErr =
"exception while saving"; return -1; } } MINIAI_API int LOADFROMTXT(void* h, const char* file) { Bot* b = get(h); if (!b || !file) return
-1; try { return doLoad(*b, file); } catch (...) { b->lastErr = "exception while loading"; return -1; } } MINIAI_API int ai_reply(void* h,
const char* input, char* out, int outLen) { Bot* b = get(h); if (!b || !input) return -1; try { double conf; int idx = predict(*b, input,
conf); b->lastConf = conf; const std::string& r = (idx >= 0 && conf >= (b->threshold >= 0 ? b->threshold : (b->mode == 1 ? 0.4 : 0.6))) ?
b->responses[idx] : b->fallback; if (out && outLen > 0) { int n = std::min((int)r.size(), outLen - 1); std::memcpy(out, r.data(),
(size_t)n); out[n] = '\0'; } return (int)r.size(); } catch (...) { return -1; } } MINIAI_API double ai_last_confidence(void* h) { Bot* b =
get(h); return b ? b->lastConf : 0.0; } MINIAI_API double ai_accuracy(void* h) { Bot* b = get(h); return b ? accuracy(*b) : -1.0; }
MINIAI_API void ai_set_threshold(void* h, double t) { if (Bot* b = get(h)) b->threshold = t; } MINIAI_API void ai_set_fallback(void* h,
const char* s) { if (Bot* b = get(h)) if (s) b->fallback = s; } MINIAI_API void ai_set_verbose(void* h, int v) { if (Bot* b = get(h))
b->verbose = v != 0; } MINIAI_API void ai_set_seed(void* h, unsigned s) { if (Bot* b = get(h)) b->seed = s; } MINIAI_API int
ai_num_examples(void* h) { Bot* b = get(h); return b ? (int)b->inputs.size() : -1; } MINIAI_API int ai_num_responses(void* h) { Bot* b =
get(h); return b ? (int)b->responses.size() : -1; } MINIAI_API int ai_vocab_size(void* h) { Bot* b = get(h); return b ?
(int)b->vocabList.size() : -1; } MINIAI_API int ai_is_trained(void* h) { Bot* b = get(h); return b ? b->mode : -1; } MINIAI_API const char*
ai_last_error(void* h) { Bot* b = get(h); return b ? b->lastErr.c_str() : "null handle"; }
