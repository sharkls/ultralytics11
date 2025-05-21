#include "log.hpp"

// #define GLOG_USE_GLOG_EXPORT
// #include <glog/logging.h>

int main(int argc, char* argv[]) {
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging("0000");
    google::SetLogDestination(google::ERROR, "");
    google::SetLogDestination(google::WARNING, "");
    google::SetLogDestination(google::FATAL, "");
    TINFO << "1111";
    TWARN << "2222";
    TERROR << "3333";
    // TFATAL << "4444";

    // LOG(INFO) << "I am INFO!";
	// LOG(WARNING) << "I am WARNING!";
	// LOG(ERROR) << "I am ERROR!";
	// LOG(FATAL) << "I am FATAL!";

    return 0;
}