#pragma once

// CLI subcommands. Each takes the arguments after the command name and
// returns a process exit code.

#include <span>
#include <string_view>

namespace engine::cli {

int cmd_inspect(std::span<const std::string_view> args);
int cmd_run(std::span<const std::string_view> args);

}  // namespace engine::cli
