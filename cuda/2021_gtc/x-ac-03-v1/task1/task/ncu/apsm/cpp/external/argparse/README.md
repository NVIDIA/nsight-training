<p align="center">
  <img height="100" src="https://i.imgur.com/oDXeMUQ.png" alt="argparse"/>
</p>

<p align="center">
  <img src="https://travis-ci.org/p-ranav/argparse.svg?branch=master" alt="travis"/>
  <a href="https://github.com/p-ranav/argparse/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license"/>
  </a>
  <img src="https://img.shields.io/badge/version-2.2-blue.svg?cacheSeconds=2592000" alt="version"/>
</p>

## Highlights

* Single header file
* Requires C++17
* MIT License

## Quick Start

Simply include argparse.hpp and you're good to go.

```cpp
#include <argparse/argparse.hpp>
```

To start parsing command-line arguments, create an ```ArgumentParser```. 

```cpp
argparse::ArgumentParser program("program name");
```

**NOTE:** There is an optional second argument to the `ArgumentParser` which is the program version. Example: `argparse::ArgumentParser program("libfoo", "1.9.0");`

To add a new argument, simply call ```.add_argument(...)```. You can provide a variadic list of argument names that you want to group together, e.g., ```-v``` and ```--verbose```

```cpp
program.add_argument("foo");
program.add_argument("-v", "--verbose"); // parameter packing
```

Argparse supports a variety of argument types including positional, optional, and compound arguments. Below you can see how to configure each of these types:

### Positional Arguments

Here's an example of a ***positional argument***:

```cpp
#include <argparse/argparse.hpp>

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("program name");

  program.add_argument("square")
    .help("display the square of a given integer")
    .action([](const std::string& value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }
  
  auto input = program.get<int>("square");
  std::cout << (input * input) << std::endl;

  return 0;
}
```

And running the code:

```bash
$ ./main 15
225
```

Here's what's happening:

* The ```add_argument()``` method is used to specify which command-line options the program is willing to accept. In this case, I’ve named it square so that it’s in line with its function.
* Command-line arguments are strings. Inorder to square the argument and print the result, we need to convert this argument to a number. In order to do this, we use the ```.action``` method and provide a lambda function that tries to convert user input into an integer.
* We can get the value stored by the parser for a given argument using ```parser.get<T>(key)``` method. 

### Optional Arguments

Now, let's look at ***optional arguments***. Optional arguments start with ```-``` or ```--```, e.g., ```--verbose``` or ```-a```. Optional arguments can be placed anywhere in the input sequence. 


```cpp
argparse::ArgumentParser program("test");

program.add_argument("--verbose")
  .help("increase output verbosity")
  .default_value(false)
  .implicit_value(true);

try {
  program.parse_args(argc, argv);
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

if (program["--verbose"] == true) {
    std::cout << "Verbosity enabled" << std::endl;
}
```

```bash
$ ./main --verbose
Verbosity enabled
```

Here's what's happening: 
* The program is written so as to display something when --verbose is specified and display nothing when not.
* Since the argument is actually optional, no error is thrown when running the program without ```--verbose```. Note that by using ```.default_value(false)```, if the optional argument isn’t used, it's value is automatically set to false. 
* By using ```.implicit_value(true)```, the user specifies that this option is more of a flag than something that requires a value. When the user provides the --verbose option, it's value is set to true. 

#### Requiring optional arguments

There are scenarios where you would like to make an optional argument ***required***. As discussed above, optional arguments either begin with `-` or `--`. You can make these types of arguments required like so:

```cpp
program.add_argument("-o", "--output")
  .required()
  .help("specify the output file.");
```

If the user does not provide a value for this parameter, an exception is thrown. 

Alternatively, you could provide a default value like so:

```cpp
program.add_argument("-o", "--output")
  .default_value(std::string("-"))
  .required()
  .help("specify the output file.");
```

#### Accessing optional arguments without default values

If you require an optional argument to be present but have no good default value for it, you can combine testing and accessing the argument as following:

```cpp
if (auto fn = program.present("-o")) {
    do_something_with(*fn);
}
```

Similar to `get`, the `present` method also accepts a template argument.  But rather than returning `T`, `parser.present<T>(key)` returns `std::optional<T>`, so that when the user does not provide a value to this parameter, the return value compares equal to `std::nullopt`.

### Negative Numbers

Optional arguments start with ```-```. Can ```argparse``` handle negative numbers? The answer is yes!

```cpp
argparse::ArgumentParser program;

program.add_argument("integer")
  .help("Input number")
  .action([](const std::string& value) { return std::stoi(value); });
  
program.add_argument("floats")
  .help("Vector of floats")
  .nargs(4)
  .action([](const std::string& value) { return std::stof(value); });

try {
  program.parse_args(argc, argv);
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

// Some code to print arguments
```

```bash
$ ./main -5 -1.1 -3.1415 -3.1e2 -4.51329E3
integer : -5
floats  : -1.1 -3.1415 -310 -4513.29
```

As you can see here, ```argparse``` supports negative integers, negative floats and scientific notation. 

### Combining Positional and Optional Arguments

```cpp
argparse::ArgumentParser program("test");

program.add_argument("square")
  .help("display the square of a given number")
  .action([](const std::string& value) { return std::stoi(value); });

program.add_argument("--verbose")
  .default_value(false)
  .implicit_value(true);

try {
  program.parse_args(argc, argv);
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

int input = program.get<int>("square");

if (program["--verbose"] == true) {
  std::cout << "The square of " << input << " is " << (input * input) << std::endl;
}
else {
  std::cout << (input * input) << std::endl;
}
```

```bash
$ ./main 4
16

$ ./main 4 --verbose
The square of 4 is 16

$ ./main --verbose 4
The square of 4 is 16
```

### Printing Help

`std::cout << program` prints a help message, including the program usage and information about the arguments registered with the `ArgumentParser`. For the previous example, here's the default help message:

```
$ ./main --help
Usage: ./main [options] square

Positional arguments:
square         display a square of a given number

Optional arguments:
-h, --help     show this help message and exit
-v, --verbose  enable verbose logging
```

You may also get the help message in string via `program.help().str()`.

### List of Arguments

ArgumentParser objects usually associate a single command-line argument with a single action to be taken. The ```.nargs``` associates a different number of command-line arguments with a single action. When using ```nargs(N)```, N arguments from the command line will be gathered together into a list.

```cpp
argparse::ArgumentParser program("main");

program.add_argument("--input_files")
  .help("The list of input files")
  .nargs(2);

try {
  program.parse_args(argc, argv);   // Example: ./main --input_files config.yml System.xml
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

auto files = program.get<std::vector<std::string>>("--input_files");  // {"config.yml", "System.xml"}
```

```ArgumentParser.get<T>()``` has specializations for ```std::vector``` and ```std::list```. So, the following variant, ```.get<std::list>```, will also work. 

```cpp
auto files = program.get<std::list<std::string>>("--input_files");  // {"config.yml", "System.xml"}
```

Using ```.action```, one can quickly build a list of desired value types from command line arguments. Here's an example:

```cpp
argparse::ArgumentParser program("main");

program.add_argument("--query_point")
  .help("3D query point")
  .nargs(3)
  .default_value(std::vector<double>{0.0, 0.0, 0.0})
  .action([](const std::string& value) { return std::stod(value); });

try {
  program.parse_args(argc, argv); // Example: ./main --query_point 3.5 4.7 9.2
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

auto query_point = program.get<std::vector<double>>("--query_point");  // {3.5, 4.7, 9.2}
```

### Compound Arguments

Compound arguments are optional arguments that are combined and provided as a single argument. Example: ```ps -aux```

```cpp
argparse::ArgumentParser program("test");

program.add_argument("-a")
  .default_value(false)
  .implicit_value(true);

program.add_argument("-b")
  .default_value(false)
  .implicit_value(true);

program.add_argument("-c")
  .nargs(2)
  .default_value(std::vector<float>{0.0f, 0.0f})
  .action([](const std::string& value) { return std::stof(value); });

try {
  program.parse_args(argc, argv);                  // Example: ./main -abc 1.95 2.47
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

auto a = program.get<bool>("-a");                  // true
auto b = program.get<bool>("-b");                  // true
auto c = program.get<std::vector<float>>("-c");    // {1.95, 2.47}

/// Some code that prints parsed arguments
```

```bash
$ ./main -ac 3.14 2.718
a = true
b = false
c = {3.14, 2.718}

$ ./main -cb
a = false
b = true
c = {0.0, 0.0}
```

Here's what's happening:
* We have three optional arguments ```-a```, ```-b``` and ```-c```. 
* ```-a``` and ```-b``` are toggle arguments.
* ```-c``` requires 2 floating point numbers from the command-line.
* argparse can handle compound arguments, e.g., ```-abc``` or ```-bac``` or ```-cab```. This only works with short single-character argument names. 
  - ```-a``` and ```-b``` become true.
  - argv is further parsed to identify the inputs mapped to ```-c```.
  - If argparse cannot find any arguments to map to c, then c defaults to {0.0, 0.0} as defined by ```.default_value```

### Gathering Remaining Arguments

`argparse` supports gathering "remaining" arguments at the end of the command, e.g., for use in a compiler:

```bash
$ compiler file1 file2 file3
```

To enable this, simply create an argument and mark it as `remaining`. All remaining arguments passed to argparse are gathered here.

```cpp
argparse::ArgumentParser program("compiler");                                                             
                                                                                                          
program.add_argument("files")                                                                             
  .remaining();                                                                                           
                                                                                                          
try {                                                                                                     
  program.parse_args(argc, argv);                                                                         
}                                                                                                         
catch (const std::runtime_error& err) {                                                                   
  std::cout << err.what() << std::endl;                                                                   
  std::cout << program;                                                                                   
  exit(0);                                                                                                
}                                                                                                         

try {
  auto files = program.get<std::vector<std::string>>("files");
  std::cout << files.size() << " files provided" << std::endl;                                           
  for (auto& file : files)                                                             
    std::cout << file << std::endl;
} catch (std::logic_error& e) {
  std::cout << "No files provided" << std::endl;
}                                                                                                  
```

When no arguments are provided:

```bash
$ ./compiler
No files provided
```

and when multiple arguments are provided:

```bash
$ ./compiler foo.txt bar.txt baz.txt
3 files provided
foo.txt
bar.txt
baz.txt
```

The process of gathering remaining arguments plays nicely with optional arguments too:

```cpp
argparse::ArgumentParser program("compiler");                                                             

program.add_arguments("-o")
  .default_value(std::string("a.out"));

program.add_argument("files")                                                                             
  .remaining();                                                                                           
                                                                                                          
try {                                                                                                     
  program.parse_args(argc, argv);                                                                         
}                                                                                                         
catch (const std::runtime_error& err) {                                                                   
  std::cout << err.what() << std::endl;                                                                   
  std::cout << program;                                                                                   
  exit(0);                                                                                                
}

auto output_filename = program.get<std::string>("-o");
std::cout << "Output filename: " << output_filename << std::endl;

try {
  auto files = program.get<std::vector<std::string>>("files");
  std::cout << files.size() << " files provided" << std::endl;                                           
  for (auto& file : files)                                                             
    std::cout << file << std::endl;
} catch (std::logic_error& e) {
  std::cout << "No files provided" << std::endl;
}

```

```bash
$ ./compiler -o main foo.cpp bar.cpp baz.cpp
Output filename: main
3 files provided
foo.cpp
bar.cpp
baz.cpp
```

***NOTE***: Remember to place all optional arguments BEFORE the remaining argument. If the optional argument is placed after the remaining arguments, it too will be deemed remaining:

```bash
$ ./compiler foo.cpp bar.cpp baz.cpp -o main
5 arguments provided
foo.cpp
bar.cpp
baz.cpp
-o
main
```

### Parent Parsers

Sometimes, several parsers share a common set of arguments. Rather than repeating the definitions of these arguments, a single parser with all the common arguments can be added as a parent to another ArgumentParser instance. The ```.add_parents``` method takes a list of ArgumentParser objects, collects all the positional and optional actions from them, and adds these actions to the ArgumentParser object being constructed:

```cpp
argparse::ArgumentParser parent_parser("main");
parent_parser.add_argument("--parent")
  .default_value(0)
  .action([](const std::string& value) { return std::stoi(value); });

argparse::ArgumentParser foo_parser("foo");
foo_parser.add_argument("foo");
foo_parser.add_parents(parent_parser);
foo_parser.parse_args({ "./main", "--parent", "2", "XXX" });   // parent = 2, foo = XXX

argparse::ArgumentParser bar_parser("bar");
bar_parser.add_argument("--bar");
bar_parser.parse_args({ "./main", "--bar", "YYY" });           // bar = YYY
```

Note You must fully initialize the parsers before passing them via ```.add_parents```. If you change the parent parsers after the child parser, those changes will not be reflected in the child.

## Further Examples

### Construct a JSON object from a filename argument

```cpp
argparse::ArgumentParser program("json_test");

program.add_argument("config")
  .action([](const std::string& value) {
    // read a JSON file
    std::ifstream stream(value);
    nlohmann::json config_json;
    stream >> config_json;
    return config_json;
  });

try {
  program.parse_args({"./test", "config.json"});
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

nlohmann::json config = program.get<nlohmann::json>("config");
```

### Positional Arguments with Compound Toggle Arguments

```cpp
argparse::ArgumentParser program("test");

program.add_argument("numbers")
  .nargs(3)
  .action([](const std::string& value) { return std::stoi(value); });

program.add_argument("-a")
  .default_value(false)
  .implicit_value(true);

program.add_argument("-b")
  .default_value(false)
  .implicit_value(true);

program.add_argument("-c")
  .nargs(2)
  .action([](const std::string& value) { return std::stof(value); });

program.add_argument("--files")
  .nargs(3);

try {
  program.parse_args(argc, argv);
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

auto numbers = program.get<std::vector<int>>("numbers");        // {1, 2, 3}
auto a = program.get<bool>("-a");                               // true
auto b = program.get<bool>("-b");                               // true
auto c = program.get<std::vector<float>>("-c");                 // {3.14f, 2.718f}
auto files = program.get<std::vector<std::string>>("--files");  // {"a.txt", "b.txt", "c.txt"}

/// Some code that prints parsed arguments
```

```bash
$ ./main 1 2 3 -abc 3.14 2.718 --files a.txt b.txt c.txt
numbers = {1, 2, 3}
a = true
b = true
c = {3.14, 2.718}
files = {"a.txt", "b.txt", "c.txt"}
```

### Restricting the set of values for an argument

```cpp
argparse::ArgumentParser program("test");

program.add_argument("input")
  .default_value("baz")
  .action([](const std::string& value) {
    static const std::vector<std::string> choices = { "foo", "bar", "baz" };
    if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
      return value;
    }
    return std::string{ "baz" };
  });

try {
  program.parse_args(argc, argv);
}
catch (const std::runtime_error& err) {
  std::cout << err.what() << std::endl;
  std::cout << program;
  exit(0);
}

auto input = program.get("input");
std::cout << input << std::endl;
```

```bash
$ ./main fex
baz
```

## Supported Toolchains

| Compiler             | Standard Library | Test Environment   |
| :------------------- | :--------------- | :----------------- |
| GCC >= 8.3.0         | libstdc++        | Ubuntu 18.04       |
| Clang >= 7.0.0       | libc++           | Xcode 10.2         |
| MSVC >= 14.16        | Microsoft STL    | Visual Studio 2017 |

## Contributing
Contributions are welcome, have a look at the [CONTRIBUTING.md](CONTRIBUTING.md) document for more information.

## Contributors ✨

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/svanveen"><img src="https://avatars1.githubusercontent.com/u/23560108?s=400&v=4" width="100px;" alt="svanveen"/><br /><sub><b>svanveen</b></sub></a></td>
    <td align="center"><a href="https://github.com/lichray"><img src="https://avatars0.githubusercontent.com/u/433009?s=400&v=4" width="100px;" alt="Zhihao Yuan"/><br /><sub><b>Zhihao Yuan</b></sub></a></td>
    <td align="center"><a href="https://github.com/wtdcode"><img src="https://avatars2.githubusercontent.com/u/30623163?s=400&v=4" width="100px;" alt="Mio"/><br /><sub><b>Mio</b></sub></a></td>
    <td align="center"><a href="https://github.com/zhihaoy"><img src="https://avatars0.githubusercontent.com/u/43971430?s=400&v=4" width="100px;" alt="zhihaoy"/><br /><sub><b>zhihaoy</b></sub></a></td>
    <td align="center"><a href="https://github.com/Jackojc"><img src="https://avatars1.githubusercontent.com/u/4932816?s=400&v=4" width="100px;" alt="Jack Clarke"/><br /><sub><b>Jack Clarke</b></sub></a></td>
    <td align="center"><a href="https://github.com/SuperWig"><img src="https://avatars2.githubusercontent.com/u/2692096?s=400&v=4" width="100px;" alt="Daniel Marshall"/><br /><sub><b>Daniel Marshall</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/MU001999"><img src="https://avatars3.githubusercontent.com/u/21022101?s=400&v=4" width="100px;" alt="mupp"/><br /><sub><b>mupp</b></sub></a></td>
    <td align="center"><a href="https://github.com/CrustyAuklet"><img src="https://avatars2.githubusercontent.com/u/9755578?s=400&v=4" width="100px;" alt="Ethan Slattery"/><br /><sub><b>Ethan Slattery</b></sub></a></td>
  </tr>
</table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

## License
The project is available under the [MIT](https://opensource.org/licenses/MIT) license.
