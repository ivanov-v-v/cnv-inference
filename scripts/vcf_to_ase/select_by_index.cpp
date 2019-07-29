#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::endl;

const size_t BUFF_SIZE = 1 << 16;
const size_t INF = std::numeric_limits<size_t>::max();

void flush_buffer(std::ofstream& outfile, std::vector<std::string>& outbuff) {
    std::string chunk = "";
    for (auto& s : outbuff) {
        chunk += s + "\n";
    }
    outfile << chunk;
    outbuff.clear();
}

int main(int argc, char** argv) {
    const char* in_path = argv[1];
    const char* id_path = argv[2];
    const char* out_path = argv[3];

    std::ifstream idfile(id_path);
    std::vector<int> id_vec { std::istream_iterator<int>(idfile), {} };
    idfile.close();

    auto it_to_id = id_vec.begin();
    std::ifstream infile(in_path);
    std::string dataline;
    std::ofstream outfile(out_path);
    std::vector<std::string> outbuff; 

    for (size_t i = 1; !infile.eof() && it_to_id != id_vec.end(); ++i) {
        infile >> dataline;
        if (i == *it_to_id) {
            ++it_to_id;
            outbuff.push_back(std::move(dataline));
            if (outbuff.size() == BUFF_SIZE) {
                flush_buffer(outfile, outbuff);
            }
        }
    }
    
    if (!outbuff.empty()) {
        flush_buffer(outfile, outbuff);
    }

    infile.close();
    outfile.close();
    return 0;
}
