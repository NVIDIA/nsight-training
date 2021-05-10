#include <doctest.hpp>
#include <argparse/argparse.hpp>

using doctest::test_suite;

TEST_CASE("Parse unknown optional argument" *
          test_suite("compound_arguments")) {

  argparse::ArgumentParser bfm("bfm");

  bfm.add_argument("-l","--load")
    .help("load a VMM into the kernel");

  bfm.add_argument("-x", "--start")
    .default_value(false)
    .implicit_value(true)
    .help("start a previously loaded VMM");

  bfm.add_argument("-d", "--dump")
    .default_value(false)
    .implicit_value(true)
    .help("output the contents of the VMM's debug buffer");

  bfm.add_argument("-s", "--stop")
    .default_value(false)
    .implicit_value(true)
    .help("stop a previously started VMM");

  bfm.add_argument("-u", "--unload")
    .default_value(false)
    .implicit_value(true)
    .help("unload a previously loaded VMM");

  bfm.add_argument("-m", "--mem")
    .default_value(64ULL)
    .action([](const std::string& val) { return std::stoull(val); })
    .help("memory in MB to give the VMM when loading");

  REQUIRE_THROWS(bfm.parse_args({ "./test.exe", "-om" }));
}
