#pragma once

// Where models come from: turning a user-supplied link into a download URL,
// checking a GGUF header before committing to a multi-GB download, and the
// one "can DynaLM run this file?" verdict shared by `list` and `pull`.

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "dynacore/base/status.h"
#include "loader/gguf/gguf.h"
#include "common/core.h"

namespace dynalm {

// Accepts
//   https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>.gguf  (as is)
//   https://huggingface.co/<owner>/<repo>/blob/<rev>/<path>.gguf     (page link -> resolve)
//   <owner>/<repo>/<path>.gguf                                        (Hugging Face, main)
//   any other http(s) URL                                              (as is)
// and returns the direct download URL. A Hugging Face repository link without
// a file is rejected with a hint, since a repo holds many quantizations.
Result<std::string> resolve_model_url(std::string_view ref);

// Last path segment of a URL, without query or fragment ("" if none).
std::string url_file_name(std::string_view url);

// Reads `general.architecture` from the first bytes of a GGUF file. Corrupt
// if the bytes are not a GGUF (e.g. an HTML error page); NotFound if the key
// is not within `head` (the caller fetched too little).
Result<std::string> peek_gguf_architecture(std::span<const std::byte> head);

// The header keys a support check needs, as far as `head` reaches. The
// tokenizer keys usually precede the (large) token list, so 256 KiB holds
// them; has_* tells whether each was seen.
// The overall verdict `dynalm inspect` prints from its three separate checks:
// "YES" only when the architecture, the tokenizer and the weights are all
// supported; otherwise "NO (<the failing parts>)", or, when the weights are
// not present (a directory with only config and tokenizer files), "PARTIAL
// (...)": never YES without having checked everything.
enum class WeightsCheck { kSupported, kUnsupported, kAbsent };
std::string support_verdict(bool architecture_ok, bool tokenizer_ok, WeightsCheck weights);

struct GgufPeek {
  std::string architecture, tokenizer_model, tokenizer_pre;
  bool has_architecture = false, has_tokenizer_model = false, has_tokenizer_pre = false;
};
Result<GgufPeek> peek_gguf_header(std::span<const std::byte> head);

// "ok" when the engine can load and run the file, otherwise the reason, e.g.
// "unsupported architecture 'qwen35'", "unsupported tensor types: iq3_s, iq1_s"
// or "unsupported tokenizer: pre-tokenizer 'x' is not supported yet".
std::string gguf_support_status(const gguf::GgufFile& f);

}  // namespace dynalm
