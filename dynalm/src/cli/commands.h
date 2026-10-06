#pragma once

// CLI subcommands. Each takes the arguments after the command name and
// returns a process exit code.

#include <span>
#include <string>
#include <string_view>

#include "common/core.h"
#include "dynacore/base/status.h"

namespace dynalm::cli {

int cmd_inspect(std::span<const std::string_view> args);
int cmd_run(std::span<const std::string_view> args);
int cmd_serve(std::span<const std::string_view> args);
int cmd_benchmark(std::span<const std::string_view> args);
int cmd_pull(std::span<const std::string_view> args);
int cmd_rm(std::span<const std::string_view> args);
int cmd_models(std::span<const std::string_view> args);
int cmd_stop(std::span<const std::string_view> args);
int cmd_doctor(std::span<const std::string_view> args);
int cmd_config(std::span<const std::string_view> args);

// Path of the model a reference names (file, directory or registry name).
// A registry name that is not downloaded yet is pulled first when
// `pull_if_missing`; otherwise it is a kNotFound naming the pull command.
Result<std::string> ensure_model(std::string_view ref, bool pull_if_missing);

}  // namespace dynalm::cli
