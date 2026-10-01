#include "runtime/text_stream.h"

#include <algorithm>

namespace engine {

TextStreamer::TextStreamer(const Tokenizer& tokenizer, std::vector<std::string> stop_strings)
    : tokenizer_(tokenizer), stops_(std::move(stop_strings)) {
  stops_.erase(std::remove_if(stops_.begin(), stops_.end(), [](const std::string& s) { return s.empty(); }),
               stops_.end());
}

size_t TextStreamer::held_suffix() const {
  size_t best = 0;
  for (const std::string& s : stops_) {
    const size_t max_k = std::min(pending_.size(), s.size() - 1);
    for (size_t k = max_k; k > best; --k) {
      if (pending_.compare(pending_.size() - k, k, s, 0, k) == 0) {
        best = k;
        break;
      }
    }
  }
  return best;
}

TextStreamer::Delta TextStreamer::push(TokenId id) {
  Delta d;
  if (stopped_) {
    d.stopped = true;
    return d;
  }
  piece_.clear();
  tokenizer_.decode_token(id, piece_, /*render_special=*/false);
  std::string complete;
  utf8_.push(piece_, complete);
  pending_ += complete;

  // Earliest complete stop string wins.
  size_t cut = std::string::npos;
  for (const std::string& s : stops_) cut = std::min(cut, pending_.find(s));
  if (cut != std::string::npos) {
    d.text = pending_.substr(0, cut);
    d.stopped = stopped_ = true;
    pending_.clear();
    return d;
  }
  const size_t keep = held_suffix();
  d.text = pending_.substr(0, pending_.size() - keep);
  pending_.erase(0, pending_.size() - keep);
  return d;
}

std::string TextStreamer::finish() {
  if (stopped_) return {};
  std::string tail;
  utf8_.flush(tail);
  pending_ += tail;
  std::string out;
  out.swap(pending_);
  return out;
}

}  // namespace engine
