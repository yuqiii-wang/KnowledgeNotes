protoc -I=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` rpc_hello.proto
protoc -I=. --cpp_out=. rpc_hello.proto