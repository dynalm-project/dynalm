#include "dynacore/lang/lang.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <memory>
#include <optional>
#include <set>

namespace dynacore::lang {
namespace {

// --- lexer ----------------------------------------------------------------------

struct Loc {
  int line = 1, col = 1;
};

enum class Tok : uint8_t { kIdent, kInt, kFloat, kPunct, kEnd };

struct Token {
  Tok kind = Tok::kEnd;
  std::string text;
  Loc loc;
};

class Lexer {
 public:
  explicit Lexer(std::string_view s) : s_(s) {}

  Result<std::vector<Token>> run() {
    std::vector<Token> out;
    for (;;) {
      skip();
      Token t;
      t.loc = loc_;
      if (pos_ >= s_.size()) {
        out.push_back(t);
        return out;
      }
      const char c = s_[pos_];
      if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_' ||
                                    s_[pos_] == '.')) {
          t.text += take();
        }
        t.kind = Tok::kIdent;
      } else if (std::isdigit(static_cast<unsigned char>(c)) ||
                 (c == '.' && pos_ + 1 < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_ + 1])))) {
        bool flt = false;
        while (pos_ < s_.size()) {
          const char d = s_[pos_];
          if (std::isdigit(static_cast<unsigned char>(d))) {
            t.text += take();
          } else if (d == '.' || d == 'e' || d == 'E') {
            flt = true;
            t.text += take();
            if ((d == 'e' || d == 'E') && pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) t.text += take();
          } else {
            break;
          }
        }
        t.kind = flt ? Tok::kFloat : Tok::kInt;
      } else if (std::string_view("(){}[]<>,:=@+*-;/").find(c) != std::string_view::npos) {
        t.text = take();
        t.kind = Tok::kPunct;
      } else {
        return InvalidArgument(std::to_string(t.loc.line) + ":" + std::to_string(t.loc.col) + ": unexpected character '" +
                               std::string(1, c) + "'");
      }
      out.push_back(std::move(t));
      if (out.size() > 1'000'000) return InvalidArgument("program too large");
    }
  }

 private:
  char take() {
    const char c = s_[pos_++];
    if (c == '\n') {
      ++loc_.line;
      loc_.col = 1;
    } else {
      ++loc_.col;
    }
    return c;
  }
  void skip() {
    for (;;) {
      while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) take();
      if (pos_ < s_.size() && (s_[pos_] == '#' || s_.substr(pos_, 2) == "//")) {
        while (pos_ < s_.size() && s_[pos_] != '\n') take();
        continue;
      }
      return;
    }
  }
  std::string_view s_;
  size_t pos_ = 0;
  Loc loc_;
};

// --- AST --------------------------------------------------------------------------

struct Expr {
  enum class Kind : uint8_t { kName, kNumber, kCall, kBinary };
  Kind kind = Kind::kName;
  Loc loc;
  std::string name;  // kName, kCall
  double number = 0;
  bool integer = false;
  char op = 0;  // kBinary: '@' '+' '*' '-'
  std::vector<std::unique_ptr<Expr>> args;
  std::vector<std::pair<std::string, std::unique_ptr<Expr>>> kwargs;
};

struct DimAst {
  std::vector<std::string> factors;  // product of ints / names
  Loc loc;
};

struct TypeAst {
  std::string kind;   // tensor | index | weight | kv
  std::string dtype;  // empty for index
  std::vector<DimAst> dims;
  Loc loc;
};

struct Stmt {
  enum class Kind : uint8_t { kConfig, kDecl, kAssign, kOutput, kSchedule };
  Kind kind = Kind::kAssign;
  Loc loc;
  std::string word;                   // kDecl: input | weight | kv
  std::vector<std::string> names;     // decl/assign target, outputs, config names
  std::vector<std::unique_ptr<Expr>> values;  // config values, assign rhs
  TypeAst type;
  std::vector<std::pair<std::string, std::string>> directives;  // schedule
};

struct GraphAst {
  std::string name;
  Loc loc;
  std::vector<std::pair<std::string, int64_t>> params;
  std::vector<Stmt> stmts;
};

// --- parser ----------------------------------------------------------------------

class Parser {
 public:
  Parser(std::vector<Token> toks, std::string file) : t_(std::move(toks)), file_(std::move(file)) {
    // Look-ahead of up to two tokens never runs past the end.
    const Token end = t_.back();
    t_.push_back(end);
    t_.push_back(end);
  }

  Result<std::vector<GraphAst>> program() {
    std::vector<GraphAst> graphs;
    while (peek().kind != Tok::kEnd) {
      ENGINE_ASSIGN_OR_RETURN(GraphAst g, graph());
      graphs.push_back(std::move(g));
    }
    if (graphs.empty()) return err(peek().loc, "expected 'graph'");
    return graphs;
  }

 private:
  Result<GraphAst> graph() {
    GraphAst g;
    g.loc = peek().loc;
    if (!word("graph")) return err(peek().loc, "expected 'graph'");
    ENGINE_ASSIGN_OR_RETURN(g.name, ident("graph name"));
    if (punct("(")) {
      while (!punct(")")) {
        ENGINE_ASSIGN_OR_RETURN(std::string p, ident("parameter"));
        int64_t def = 1;
        if (punct("=")) {
          ENGINE_ASSIGN_OR_RETURN(def, integer());
        }
        g.params.emplace_back(p, def);
        if (!punct(",") && peek().text != ")") return err(peek().loc, "expected ',' or ')'");
      }
    }
    if (!punct("{")) return err(peek().loc, "expected '{'");
    while (!punct("}")) {
      if (peek().kind == Tok::kEnd) return err(peek().loc, "missing '}' at end of graph " + g.name);
      ENGINE_ASSIGN_OR_RETURN(Stmt s, stmt());
      g.stmts.push_back(std::move(s));
      punct(";");
    }
    return g;
  }

  Result<Stmt> stmt() {
    Stmt s;
    s.loc = peek().loc;
    const std::string w = peek().text;
    if (peek().kind == Tok::kIdent && w == "config") {
      ++i_;
      s.kind = Stmt::Kind::kConfig;
      do {
        ENGINE_ASSIGN_OR_RETURN(std::string n, ident("config name"));
        if (!punct("=")) return err(peek().loc, "expected '=' in config");
        ENGINE_ASSIGN_OR_RETURN(auto e, expr());
        s.names.push_back(n);
        s.values.push_back(std::move(e));
      } while (punct(","));
      return s;
    }
    if (peek().kind == Tok::kIdent && (w == "input" || w == "weight" || w == "kv") && t_[i_ + 1].kind == Tok::kIdent &&
        t_[i_ + 2].text == ":") {
      ++i_;
      s.kind = Stmt::Kind::kDecl;
      s.word = w;
      ENGINE_ASSIGN_OR_RETURN(std::string n, ident("name"));
      s.names.push_back(n);
      punct(":");
      ENGINE_ASSIGN_OR_RETURN(s.type, type());
      return s;
    }
    if (peek().kind == Tok::kIdent && w == "output") {
      ++i_;
      s.kind = Stmt::Kind::kOutput;
      do {
        ENGINE_ASSIGN_OR_RETURN(std::string n, ident("output name"));
        s.names.push_back(n);
      } while (punct(","));
      return s;
    }
    if (peek().kind == Tok::kIdent && w == "schedule") {
      ++i_;
      s.kind = Stmt::Kind::kSchedule;
      if (!punct("{")) return err(peek().loc, "expected '{' after schedule");
      while (!punct("}")) {
        if (peek().kind == Tok::kEnd) return err(peek().loc, "missing '}' in schedule");
        ENGINE_ASSIGN_OR_RETURN(std::string d, ident("schedule directive"));
        ENGINE_ASSIGN_OR_RETURN(std::string v, ident("directive value"));
        s.directives.emplace_back(d, v);
        punct(";");
      }
      return s;
    }
    s.kind = Stmt::Kind::kAssign;
    ENGINE_ASSIGN_OR_RETURN(std::string n, ident("statement"));
    s.names.push_back(n);
    if (!punct("=")) return err(peek().loc, "expected '=' after " + n);
    ENGINE_ASSIGN_OR_RETURN(auto e, expr());
    s.values.push_back(std::move(e));
    return s;
  }

  Result<TypeAst> type() {
    TypeAst t;
    t.loc = peek().loc;
    ENGINE_ASSIGN_OR_RETURN(std::string k, ident("type"));
    if (k == "tensor" || k == "weight" || k == "kv") {
      t.kind = k;
      if (!punct("<")) return err(peek().loc, "expected '<' after " + k);
      ENGINE_ASSIGN_OR_RETURN(t.dtype, ident("element type"));
      if (!punct(">")) return err(peek().loc, "expected '>'");
    } else if (k == "index") {
      t.kind = "index";
    } else {
      t.kind = "";  // shorthand: q4_K[...] (weight) or f32[...] (by declaration)
      t.dtype = k;
    }
    if (!punct("[")) return err(peek().loc, "expected '[' with the shape");
    while (!punct("]")) {
      DimAst d;
      d.loc = peek().loc;
      do {
        if (peek().kind == Tok::kInt || peek().kind == Tok::kIdent) {
          d.factors.push_back(t_[i_++].text);
        } else {
          return err(peek().loc, "expected a dimension (number or name)");
        }
      } while (punct("*"));
      t.dims.push_back(std::move(d));
      if (!punct(",") && peek().text != "]") return err(peek().loc, "expected ',' or ']'");
    }
    return t;
  }

  // expr := sum ; sum := prod (('+'|'-') prod)* ; prod := matm ('*' matm)* ; matm := atom ('@' atom)*
  Result<std::unique_ptr<Expr>> expr() { return sum(); }
  Result<std::unique_ptr<Expr>> sum() {
    ENGINE_ASSIGN_OR_RETURN(auto l, prod());
    while (peek().text == "+" || peek().text == "-") {
      auto b = std::make_unique<Expr>();
      b->kind = Expr::Kind::kBinary;
      b->loc = peek().loc;
      b->op = t_[i_++].text[0];
      ENGINE_ASSIGN_OR_RETURN(auto r, prod());
      b->args.push_back(std::move(l));
      b->args.push_back(std::move(r));
      l = std::move(b);
    }
    return l;
  }
  Result<std::unique_ptr<Expr>> prod() {
    ENGINE_ASSIGN_OR_RETURN(auto l, matm());
    while (peek().text == "*" || peek().text == "/") {
      auto b = std::make_unique<Expr>();
      b->kind = Expr::Kind::kBinary;
      b->loc = peek().loc;
      b->op = t_[i_++].text[0];
      ENGINE_ASSIGN_OR_RETURN(auto r, matm());
      b->args.push_back(std::move(l));
      b->args.push_back(std::move(r));
      l = std::move(b);
    }
    return l;
  }
  Result<std::unique_ptr<Expr>> matm() {
    ENGINE_ASSIGN_OR_RETURN(auto l, atom());
    while (peek().text == "@") {
      auto b = std::make_unique<Expr>();
      b->kind = Expr::Kind::kBinary;
      b->loc = peek().loc;
      b->op = '@';
      ++i_;
      ENGINE_ASSIGN_OR_RETURN(auto r, atom());
      b->args.push_back(std::move(l));
      b->args.push_back(std::move(r));
      l = std::move(b);
    }
    return l;
  }
  Result<std::unique_ptr<Expr>> atom() {
    auto e = std::make_unique<Expr>();
    e->loc = peek().loc;
    if (punct("(")) {
      ENGINE_ASSIGN_OR_RETURN(auto inner, expr());
      if (!punct(")")) return err(peek().loc, "expected ')'");
      return inner;
    }
    if (peek().text == "-" && (t_[i_ + 1].kind == Tok::kInt || t_[i_ + 1].kind == Tok::kFloat)) {
      ++i_;
      ENGINE_ASSIGN_OR_RETURN(auto n, atom());
      n->number = -n->number;
      return n;
    }
    if (peek().kind == Tok::kInt || peek().kind == Tok::kFloat) {
      e->kind = Expr::Kind::kNumber;
      e->integer = peek().kind == Tok::kInt;
      const std::string& text = t_[i_++].text;
      try {
        e->number = std::stod(text);
      } catch (...) {
        return err(e->loc, "bad number '" + text + "'");
      }
      return e;
    }
    if (peek().kind != Tok::kIdent) return err(peek().loc, "expected an expression");
    e->name = t_[i_++].text;
    if (!punct("(")) {
      e->kind = Expr::Kind::kName;
      return e;
    }
    e->kind = Expr::Kind::kCall;
    while (!punct(")")) {
      if (peek().kind == Tok::kEnd) return err(peek().loc, "missing ')' in call to " + e->name);
      if (peek().kind == Tok::kIdent && t_[i_ + 1].text == "=") {
        std::string k = t_[i_].text;
        i_ += 2;
        ENGINE_ASSIGN_OR_RETURN(auto v, expr());
        e->kwargs.emplace_back(std::move(k), std::move(v));
      } else {
        if (!e->kwargs.empty()) return err(peek().loc, "positional argument after keyword argument");
        ENGINE_ASSIGN_OR_RETURN(auto v, expr());
        e->args.push_back(std::move(v));
      }
      if (!punct(",") && peek().text != ")") return err(peek().loc, "expected ',' or ')'");
    }
    return e;
  }

  const Token& peek() const { return t_[i_]; }
  bool punct(std::string_view p) {
    if (t_[i_].kind == Tok::kPunct && t_[i_].text == p) {
      ++i_;
      return true;
    }
    return false;
  }
  bool word(std::string_view w) {
    if (t_[i_].kind == Tok::kIdent && t_[i_].text == w) {
      ++i_;
      return true;
    }
    return false;
  }
  Result<std::string> ident(std::string_view what) {
    if (t_[i_].kind != Tok::kIdent) return err(peek().loc, "expected " + std::string(what));
    return t_[i_++].text;
  }
  Result<int64_t> integer() {
    if (t_[i_].kind != Tok::kInt) return err(peek().loc, "expected an integer");
    return std::stoll(t_[i_++].text);
  }
  Status err(Loc l, const std::string& m) const {
    return InvalidArgument(file_ + ":" + std::to_string(l.line) + ":" + std::to_string(l.col) + ": " + m);
  }

  std::vector<Token> t_;
  size_t i_ = 0;
  std::string file_;
};

// --- semantic analysis and lowering to IR -------------------------------------------

class Lowering {
 public:
  Lowering(const GraphAst& g, std::string file) : g_(g), file_(std::move(file)), prog_{ir::Graph(g.name), {}, {}, {}} {}

  Result<Program> run() {
    for (const auto& [p, def] : g_.params) {
      symbols_.insert(p);
      prog_.symbols[p] = def;
    }
    ir::Builder b(prog_.graph);
    b_ = &b;
    for (const Stmt& s : g_.stmts) ENGINE_RETURN_IF_ERROR(stmt(s));
    if (prog_.outputs.empty()) return err(g_.loc, "graph " + g_.name + " has no 'output'");
    return std::move(prog_);
  }

 private:
  Status stmt(const Stmt& s) {
    switch (s.kind) {
      case Stmt::Kind::kConfig:
        for (size_t i = 0; i < s.names.size(); ++i) {
          ENGINE_ASSIGN_OR_RETURN(double v, number(*s.values[i]));
          if (consts_.count(s.names[i]) || env_.count(s.names[i])) return err(s.loc, s.names[i] + " is already defined");
          consts_[s.names[i]] = v;
        }
        return Status::Ok();
      case Stmt::Kind::kDecl: {
        const std::string& n = s.names[0];
        if (env_.count(n) || consts_.count(n)) return err(s.loc, n + " is already defined");
        ENGINE_ASSIGN_OR_RETURN(ir::Type t, type(s.type, s.word));
        ir::ValueId v;
        if (s.word == "weight") {
          v = b_->weight(n, t);
        } else if (s.word == "kv") {
          if (t.kind != ir::ValueKind::kKv || t.rank() != 4) {
            return err(s.loc, "kv " + n + " must be kv<dtype>[blocks, kv_heads, block_size, head_dim]");
          }
          v = b_->kv_cache(n, t);
        } else {
          if (t.kind == ir::ValueKind::kWeight || t.kind == ir::ValueKind::kKv) {
            return err(s.loc, "input " + n + " must be a tensor<...> or index[...]");
          }
          v = b_->input(n, t);
        }
        env_[n] = {v, s.word == "input" || s.word == "weight"};
        return Status::Ok();
      }
      case Stmt::Kind::kAssign: {
        const std::string& n = s.names[0];
        if (consts_.count(n)) return err(s.loc, n + " is a config constant");
        if (auto it = env_.find(n); it != env_.end() && it->second.readonly) {
          return err(s.loc, "cannot assign to " + n + " (declared as an input or weight)");
        }
        ENGINE_ASSIGN_OR_RETURN(ir::ValueId v, value(*s.values[0]));
        if (prog_.graph.value(v).name.empty()) prog_.graph.mutable_value(v).name = n;
        env_[n] = {v, false};
        return Status::Ok();
      }
      case Stmt::Kind::kOutput:
        for (const std::string& n : s.names) {
          auto it = env_.find(n);
          if (it == env_.end()) return err(s.loc, "output " + n + " is not defined");
          prog_.outputs.push_back(it->second.id);
        }
        return Status::Ok();
      case Stmt::Kind::kSchedule:
        for (const auto& [d, v] : s.directives) {
          if (d == "fuse" && (v == "gated" || v == "none")) {
            prog_.fusion.gated_mlp = v == "gated";
          } else if (d == "group" && (v == "shared_input" || v == "none")) {
            prog_.fusion.group_shared_input = v == "shared_input";
          } else {
            return err(s.loc, "unknown schedule directive '" + d + " " + v +
                                  "' (supported: fuse gated|none, group shared_input|none)");
          }
        }
        return Status::Ok();
    }
    return Status::Ok();
  }

  Result<ir::Type> type(const TypeAst& t, const std::string& decl) {
    ir::Shape shape;
    for (const DimAst& d : t.dims) {
      int64_t prod = 1;
      std::string sym;
      for (const std::string& f : d.factors) {
        if (std::isdigit(static_cast<unsigned char>(f[0]))) {
          prod *= std::stoll(f);
        } else if (auto c = consts_.find(f); c != consts_.end()) {
          prod *= static_cast<int64_t>(c->second);
        } else if (symbols_.count(f)) {
          if (!sym.empty() || d.factors.size() > 1) return err(d.loc, "a symbolic dimension cannot be multiplied");
          sym = f;
        } else {
          return err(d.loc, "unknown dimension '" + f + "' (declare it with config or as a graph parameter)");
        }
      }
      shape.push_back(sym.empty() ? ir::Dim::of(prod) : ir::Dim::symbol(sym));
    }
    if (t.kind == "index") return ir::index_type(shape);
    DType dt;
    if (!parse_dtype(t.dtype, dt)) return err(t.loc, "unknown element type '" + t.dtype + "'");
    if (t.kind == "kv" || decl == "kv") {
      if (shape.size() != 4 || !shape[2].known()) return err(t.loc, "kv type needs [blocks, kv_heads, block_size, head_dim]");
      return ir::kv_type(dt, shape[0].size, shape[1].size, shape[2].size, shape[3].size);
    }
    if (t.kind == "weight" || decl == "weight") return ir::weight_type(dt, shape);
    if (dtype_is_quantized(dt)) return err(t.loc, "block-quantized " + t.dtype + " is a weight type");
    return ir::tensor_type(dt, shape);
  }

  Result<double> number(const Expr& e) {
    if (e.kind == Expr::Kind::kNumber) return e.number;
    if (e.kind == Expr::Kind::kName) {
      if (auto c = consts_.find(e.name); c != consts_.end()) return c->second;
      return err(e.loc, "'" + e.name + "' is not a constant");
    }
    if (e.kind == Expr::Kind::kBinary && e.op != '@') {
      ENGINE_ASSIGN_OR_RETURN(double l, number(*e.args[0]));
      ENGINE_ASSIGN_OR_RETURN(double r, number(*e.args[1]));
      switch (e.op) {
        case '+': return l + r;
        case '-': return l - r;
        case '*': return l * r;
        case '/':
          if (r == 0) return err(e.loc, "division by zero");
          return l / r;
      }
    }
    if (e.kind == Expr::Kind::kCall && e.name == "sqrt" && e.args.size() == 1) {
      ENGINE_ASSIGN_OR_RETURN(double v, number(*e.args[0]));
      return std::sqrt(v);
    }
    return err(e.loc, "expected a constant expression");
  }

  Result<int64_t> count(const Expr& e) {
    ENGINE_ASSIGN_OR_RETURN(double v, number(e));
    if (v != std::floor(v) || v <= 0) return err(e.loc, "expected a positive integer");
    return static_cast<int64_t>(v);
  }

  // Lowers an expression to a value; checks the IR builder's verdict after.
  Result<ir::ValueId> value(const Expr& e) {
    ENGINE_ASSIGN_OR_RETURN(ir::ValueId v, lower(e));
    if (!b_->status().ok()) return err(e.loc, b_->status().message());
    return v;
  }

  Result<ir::ValueId> lower(const Expr& e) {
    switch (e.kind) {
      case Expr::Kind::kNumber:
        return err(e.loc, "a number cannot be a tensor value");
      case Expr::Kind::kName: {
        auto it = env_.find(e.name);
        if (it == env_.end()) {
          return err(e.loc, consts_.count(e.name) ? "'" + e.name + "' is a constant, not a tensor"
                                                  : "undefined name '" + e.name + "'");
        }
        return it->second.id;
      }
      case Expr::Kind::kBinary: {
        if (e.op == '+' && e.args[0]->kind == Expr::Kind::kBinary && e.args[0]->op == '@') {
          // (x @ W) + b with a [N] weight: a matmul with bias.
          const Expr& mm = *e.args[0];
          ENGINE_ASSIGN_OR_RETURN(ir::ValueId bias, lower(*e.args[1]));
          const ir::Type& bt = prog_.graph.value(bias).type;
          if (bt.kind == ir::ValueKind::kWeight && bt.rank() == 1) {
            ENGINE_ASSIGN_OR_RETURN(ir::ValueId x, lower(*mm.args[0]));
            ENGINE_ASSIGN_OR_RETURN(ir::ValueId w, weight_operand(*mm.args[1]));
            return b_->matmul(x, w, bias);
          }
        }
        if (e.op == '@') {
          ENGINE_ASSIGN_OR_RETURN(ir::ValueId x, lower(*e.args[0]));
          ENGINE_ASSIGN_OR_RETURN(ir::ValueId w, weight_operand(*e.args[1]));
          return b_->matmul(x, w);
        }
        ENGINE_ASSIGN_OR_RETURN(ir::ValueId l, lower(*e.args[0]));
        ENGINE_ASSIGN_OR_RETURN(ir::ValueId r, lower(*e.args[1]));
        if (e.op == '+') return b_->add(l, r);
        if (e.op == '*') return b_->mul(l, r);
        return err(e.loc, std::string("operator '") + e.op + "' is not defined on tensors");
      }
      case Expr::Kind::kCall:
        return call(e);
    }
    return err(e.loc, "bad expression");
  }

  Result<ir::ValueId> weight_operand(const Expr& e) {
    ENGINE_ASSIGN_OR_RETURN(ir::ValueId w, lower(e));
    if (prog_.graph.value(w).type.kind != ir::ValueKind::kWeight) {
      return err(e.loc, "the right side of '@' must be a weight ([out, in]); got " +
                            ir::type_to_string(prog_.graph.value(w).type));
    }
    return w;
  }

  const Expr* kwarg(const Expr& e, std::string_view k) {
    for (const auto& [name, v] : e.kwargs) {
      if (name == k) return v.get();
    }
    return nullptr;
  }

  Result<std::string> word_arg(const Expr& e, std::string_view k, std::string def) {
    const Expr* v = kwarg(e, k);
    if (v == nullptr) return def;
    if (v->kind != Expr::Kind::kName || env_.count(v->name) || consts_.count(v->name)) {
      return err(v->loc, std::string(k) + " must be a word");
    }
    return v->name;
  }

  Status arity(const Expr& e, size_t lo, size_t hi, std::initializer_list<std::string_view> kws) {
    if (e.args.size() < lo || e.args.size() > hi) {
      return err(e.loc, e.name + "() takes " + std::to_string(lo) + (lo == hi ? "" : "-" + std::to_string(hi)) +
                            " positional arguments, got " + std::to_string(e.args.size()));
    }
    for (const auto& [k, v] : e.kwargs) {
      if (std::find(kws.begin(), kws.end(), k) == kws.end()) return err(v->loc, e.name + "() has no argument '" + k + "'");
    }
    return Status::Ok();
  }

  Result<ir::ValueId> call(const Expr& e) {
    const std::string& f = e.name;
    auto arg = [&](size_t i) { return lower(*e.args[i]); };
    if (f == "rmsnorm" || f == "layernorm") {
      ENGINE_RETURN_IF_ERROR(arity(e, 2, f == "layernorm" ? 3 : 2, {"eps"}));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId x, arg(0));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId w, arg(1));
      double eps = f == "rmsnorm" ? 1e-6 : 1e-5;
      if (const Expr* k = kwarg(e, "eps")) {
        ENGINE_ASSIGN_OR_RETURN(eps, number(*k));
      }
      if (f == "rmsnorm") return b_->rms_norm(x, w, eps);
      ir::ValueId bias = ir::kNoValue;
      if (e.args.size() == 3) {
        ENGINE_ASSIGN_OR_RETURN(bias, arg(2));
      }
      return b_->layer_norm(x, w, bias, eps);
    }
    if (f == "rope") {
      ENGINE_RETURN_IF_ERROR(arity(e, 4, 4, {"base", "style", "dim"}));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId x, arg(0));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId pos, arg(1));
      ENGINE_ASSIGN_OR_RETURN(int64_t heads, count(*e.args[2]));
      ENGINE_ASSIGN_OR_RETURN(int64_t hd, count(*e.args[3]));
      ir::Attrs a;
      double base = 10000;
      if (const Expr* k = kwarg(e, "base")) {
        ENGINE_ASSIGN_OR_RETURN(base, number(*k));
      }
      a["base"] = base;
      ENGINE_ASSIGN_OR_RETURN(std::string style, word_arg(e, "style", "half_split"));
      if (style != "half_split" && style != "interleaved") return err(e.loc, "rope style is half_split or interleaved");
      a["style"] = style;
      int64_t dim = hd;
      if (const Expr* k = kwarg(e, "dim")) {
        ENGINE_ASSIGN_OR_RETURN(dim, count(*k));
      }
      a["dim"] = dim;
      return b_->rope(x, pos, heads, hd, std::move(a));
    }
    if (f == "kv_write") {
      ENGINE_RETURN_IF_ERROR(arity(e, 5, 5, {}));
      std::vector<ir::ValueId> in;
      for (size_t i = 0; i < 5; ++i) {
        ENGINE_ASSIGN_OR_RETURN(ir::ValueId v, arg(i));
        in.push_back(v);
      }
      return b_->kv_write(in[0], in[1], in[2], in[3], in[4]);
    }
    if (f == "attention") {
      ENGINE_RETURN_IF_ERROR(arity(e, 7, 7, {"scale", "softcap", "window"}));
      std::vector<ir::ValueId> in;
      for (size_t i = 0; i < 4; ++i) {
        ENGINE_ASSIGN_OR_RETURN(ir::ValueId v, arg(i));
        in.push_back(v);
      }
      ENGINE_ASSIGN_OR_RETURN(int64_t heads, count(*e.args[4]));
      ENGINE_ASSIGN_OR_RETURN(int64_t kvh, count(*e.args[5]));
      ENGINE_ASSIGN_OR_RETURN(int64_t hd, count(*e.args[6]));
      double scale = 1.0 / std::sqrt(static_cast<double>(hd));
      if (const Expr* k = kwarg(e, "scale")) {
        ENGINE_ASSIGN_OR_RETURN(scale, number(*k));
      }
      ir::Attrs extra;
      const ir::Type& kv = prog_.graph.value(in[1]).type;
      if (kv.layout.kind == ir::LayoutKind::kPaged) extra["block_size"] = kv.layout.param;
      if (const Expr* k = kwarg(e, "softcap")) {
        ENGINE_ASSIGN_OR_RETURN(double c, number(*k));
        extra["softcap"] = c;
      }
      if (const Expr* k = kwarg(e, "window")) {
        ENGINE_ASSIGN_OR_RETURN(int64_t w, count(*k));
        extra["window"] = w;
      }
      return b_->attention(in[0], in[1], in[2], in[3], heads, kvh, hd, scale, std::move(extra));
    }
    if (f == "silu" || f == "gelu" || f == "gelu_tanh") {
      ENGINE_RETURN_IF_ERROR(arity(e, 1, 1, {}));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId x, arg(0));
      return b_->activation(x, f);
    }
    if (f == "swiglu" || f == "geglu") {
      ENGINE_RETURN_IF_ERROR(arity(e, 2, 2, {}));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId g, arg(0));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId u, arg(1));
      return b_->act_mul(g, u, f == "swiglu" ? "silu" : "gelu_tanh");
    }
    if (f == "softmax") {
      ENGINE_RETURN_IF_ERROR(arity(e, 1, 1, {}));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId x, arg(0));
      return b_->softmax(x);
    }
    if (f == "scale" || f == "softcap") {
      ENGINE_RETURN_IF_ERROR(arity(e, 2, 2, {}));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId x, arg(0));
      ENGINE_ASSIGN_OR_RETURN(double v, number(*e.args[1]));
      if (f == "scale") return b_->scale(x, v);
      return b_->generic(ir::OpKind::kSoftcap, {x}, {{"cap", v}});
    }
    if (f == "embedding") {
      ENGINE_RETURN_IF_ERROR(arity(e, 2, 2, {}));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId t, arg(0));
      ENGINE_ASSIGN_OR_RETURN(ir::ValueId ids, arg(1));
      return b_->embedding(t, ids);
    }
    return err(e.loc, "unknown function '" + f +
                          "' (rmsnorm, layernorm, rope, kv_write, attention, silu, gelu, gelu_tanh, swiglu, geglu, "
                          "softmax, scale, softcap, embedding)");
  }

  Status err(Loc l, const std::string& m) const {
    return InvalidArgument(file_ + ":" + std::to_string(l.line) + ":" + std::to_string(l.col) + ": " + m);
  }

  struct Binding {
    ir::ValueId id;
    bool readonly;
  };
  const GraphAst& g_;
  std::string file_;
  Program prog_;
  ir::Builder* b_ = nullptr;
  std::map<std::string, Binding> env_;
  std::map<std::string, double> consts_;
  std::set<std::string> symbols_;
};

}  // namespace

Result<Program> compile(std::string_view source, std::string_view file, std::string_view graph_name) {
  if (source.size() > (16u << 20)) return InvalidArgument(std::string(file) + ": source larger than 16 MiB");
  Result<std::vector<Token>> lexed = Lexer(source).run();
  if (!lexed.ok()) return InvalidArgument(std::string(file) + ":" + lexed.status().message());
  std::vector<Token> toks = std::move(*lexed);
  Parser parser(std::move(toks), std::string(file));
  ENGINE_ASSIGN_OR_RETURN(std::vector<GraphAst> graphs, parser.program());
  const GraphAst* chosen = &graphs.front();
  if (!graph_name.empty()) {
    chosen = nullptr;
    for (const GraphAst& g : graphs) {
      if (g.name == graph_name) chosen = &g;
    }
    if (chosen == nullptr) return NotFound(std::string(file) + ": no graph named '" + std::string(graph_name) + "'");
  }
  return Lowering(*chosen, std::string(file)).run();
}

}  // namespace dynacore::lang
