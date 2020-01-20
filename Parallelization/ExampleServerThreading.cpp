/*This is a sample of a previous commit of my personal project custom file server.*/

#include <map>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>
#include <grpcpp/grpcpp.h>

#include "dfs-service.grpc.pb.h"
#include "dfslibx-call-data.h"
#include "dfslibx-service-runner.h"
#include "shared.h"
#include "dfslib-servernode.h"

#include <google/protobuf/util/time_util.h>

using grpc::Status;
using grpc::Server;
using grpc::StatusCode;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::ServerContext;
using grpc::ServerBuilder;
using grpc::string_ref;

using dfs_service::DFSService;
using namespace std;

using dfs_service::empty;
using dfs_service::files;
using dfs_service::file;
using dfs_service::fileStatus;
using dfs_service::fileData;
using dfs_service::fileAck;

using std::chrono::milliseconds;
using std::chrono::system_clock;
using google::protobuf::RepeatedPtrField;
using google::protobuf::util::TimeUtil;
using google::protobuf::Timestamp;
using google::protobuf::uint64;


using FileRequestType = dfs_service::file;
using FileListResponseType = dfs_service::files;

extern dfs_log_level_e DFS_LOG_LEVEL;


class DFSServiceImpl final :
    public DFSService::WithAsyncMethod_CallbackList<DFSService::Service>,
        public DFSCallDataManager<FileRequestType , FileListResponseType> {

private:

    /** The runner service **/
    DFSServiceRunner<FileRequestType, FileListResponseType> runner;

    /** The mount path **/
    std::string mount_path;

    /** Mutex**/
    std::mutex queue_mutex;

    /** The vector of tags for synchronization **/
    std::vector<QueueRequest<FileRequestType, FileListResponseType>> queued_tags;


    const std::string WrapPath(const std::string &filepath) {
        return this->mount_path + filepath;
    }

    /** CRC Table **/
    CRC::Table<std::uint32_t, 32> crc_table;

public:

    DFSServiceImpl(const std::string& mount_path, const std::string& server_address):
        mount_path(mount_path), crc_table(CRC::CRC_32()) {

        this->runner.SetService(this);
        this->runner.SetAddress(server_address);
        this->runner.SetQueuedRequestsCallback([&]{ this->ProcessQueuedRequests(); });

        DIR *dir;
        if ((dir = opendir(mount_path.c_str())) == NULL) {
            dfs_log(LL_ERROR) << "Failed to open directory." << mount_path;
            exit(EXIT_FAILURE);
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            struct stat pathStatus;
            string dirEntry(entry->d_name);
            string path = WrapPath(dirEntry);
            stat(path.c_str(), &pathStatus);

            if (!S_ISREG(pathStatus.st_mode)){
                continue;
            }
            fileNameToMutexMap[dirEntry] = make_unique<shared_timed_mutex>();
        }
        closedir(dir);

    }

    ~DFSServiceImpl() {
        this->runner.Shutdown();
    }

    void Run() {
        this->runner.Run();
    }


    Status performChecksum(const multimap<string_ref, string_ref>& metadata, const string& path) {
        auto checksumGetter = metadata.find(checksumKey);
        if (checksumGetter == metadata.end()){
            stringstream ss;
            return Status(StatusCode::INTERNAL, ss.str());
        }
        unsigned long clientHash = stoul(string(checksumGetter->second.begin(), checksumGetter->second.end()));
        unsigned long serverHash = dfs_file_checksum(path, &crc_table);

        if (clientHash == serverHash) {
            stringstream ss;
            return Status(StatusCode::ALREADY_EXISTS, ss.str());
        }
        return Status::OK; // Not Already Existing
    }

    shared_timed_mutex serverMutex;

    shared_timed_mutex fileNameToClientIdAccess;
    map<string, string> fileNameToClientIdMap;

    shared_timed_mutex fileNameToMutexAccess;
    map<string, unique_ptr<shared_timed_mutex>> fileNameToMutexMap;

    void releaseMutex(string fileName) {
        fileNameToClientIdAccess.lock();
        fileNameToClientIdMap.erase(fileName);
        fileNameToClientIdAccess.unlock();
    }

    shared_timed_mutex* getFileAccessMutex(string fileName) {
        fileNameToMutexAccess.lock_shared();
        auto fileAccessMutexV = fileNameToMutexMap.find(fileName);
        shared_timed_mutex* fileAccessMutex = fileAccessMutexV->second.get();
        fileNameToMutexAccess.unlock_shared();
        return fileAccessMutex;
    }

    void addFileAccessMutex(string fileName) {
        fileNameToMutexAccess.lock();
        if (fileNameToMutexMap.find(fileName) == fileNameToMutexMap.end()){
            fileNameToMutexMap[fileName] = make_unique<shared_timed_mutex>();
        }
        fileNameToMutexAccess.unlock();
    }

    Status getWriteLock(ServerContext* context, const file* request, dfs_service::empty * response) override {
        
        const multimap<string_ref, string_ref>& metadata = context->client_metadata();
        auto clientIdGetter = metadata.find(clientKey);

        if (clientIdGetter == metadata.end()){ // Can't find
            stringstream ss;
            ss << "Missing client key" << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::INTERNAL, ss.str());
        }
        auto clientId = string(clientIdGetter->second.begin(), clientIdGetter->second.end());

        fileNameToClientIdAccess.lock();
        auto clientIdLockGetter = fileNameToClientIdMap.find(request->name());
        
        string lockClientId;

        if (clientIdLockGetter != fileNameToClientIdMap.end() && (lockClientId = clientIdLockGetter->second).compare(clientId) != 0){
            fileNameToClientIdAccess.unlock();

            stringstream ss;
            ss << "Lock already in use";
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::RESOURCE_EXHAUSTED, ss.str());

        } else if (lockClientId.compare(clientId) == 0) { // Already own lock
            fileNameToClientIdAccess.unlock();
            return Status::OK;
        }

        fileNameToClientIdMap[request->name()] = clientId;
        fileNameToClientIdAccess.unlock();


        addFileAccessMutex(request->name()); // Subject File to Access Mutex

        return Status::OK;
    }

    Status getFile(ServerContext* context, const file* request, ServerWriter<fileData>* writer) override {

        const string& path = WrapPath(request->name());

        struct stat status;

        addFileAccessMutex(request->name()); //Make sure file is subjected to mutex
        shared_timed_mutex* fileAccessMutex = getFileAccessMutex(request->name());

        fileAccessMutex->lock();

        if (stat(path.c_str(), &status) != 0){
            stringstream ss;
            ss << "File does not exist: " << path << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::NOT_FOUND, ss.str());
        }

        Status checksumCode = performChecksum(context->client_metadata(), path);
        if (!checksumCode.ok()){

            if (checksumCode.error_code() == StatusCode::ALREADY_EXISTS) {
                const multimap<string_ref, string_ref>& metadata = context->client_metadata();
                auto mtimeGetter = metadata.find(mtimeKey);
                if (mtimeGetter == metadata.end()){
                    fileAccessMutex->unlock();

                    stringstream ss;
                    ss << "Missing MTime." << endl;
                    dfs_log(LL_ERROR) << ss.str();
                    return Status(StatusCode::INTERNAL, ss.str());
                }
                long mtime = stol(string(mtimeGetter->second.begin(), mtimeGetter->second.end()));

                if (mtime > status.st_mtime){
                    dfs_log(LL_SYSINFO) << "Need to update MTime";
                    struct utimbuf ub;
                    ub.modtime = mtime;
                    ub.actime = mtime;
                    if (!utime(path.c_str(), &ub)) {
                        dfs_log(LL_SYSINFO) << "Updated MTime";
                    }
                }
            }

            fileAccessMutex->unlock();
            dfs_log(LL_ERROR) << checksumCode.error_message();
            return checksumCode;
        }

        fileAccessMutex->unlock();

        fileAccessMutex->lock_shared();

        int fileSize = status.st_size;
        
        ifstream reader(path);

        fileData data;

        try {
            int bytesSent = 0;

            while(!reader.eof() && bytesSent < fileSize){
                int bytesToSend = min(fileSize - bytesSent, dataSize);
                char buffer[dataSize];

                if (context->IsCancelled()){
                    fileAccessMutex->unlock_shared();
                    const string& err = "Deadline Exceeded.";
                    dfs_log(LL_ERROR) << err;
                    return Status(StatusCode::DEADLINE_EXCEEDED, err);
                }
                reader.read(buffer, bytesToSend);
                data.set_contents(buffer, bytesToSend);
                writer->Write(data);
                dfs_log(LL_SYSINFO) << "Read " << bytesToSend << " bytes";
                bytesSent += bytesToSend;
            }
            reader.close();
            fileAccessMutex->unlock_shared();

            if (bytesSent != fileSize) {
                stringstream ss;
                ss << "Error send." << endl;
                return Status(StatusCode::INTERNAL, ss.str());
            }

        } catch (exception const& err) {
            fileAccessMutex->unlock_shared();
            stringstream ss;
            ss << "Error reading file: " << err.what() << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::INTERNAL, ss.str());
        }

        return Status::OK;
    }

    Status writeFile(ServerContext* context, ServerReader<fileData>* reader, fileAck* response) override {

        const multimap<string_ref, string_ref>& metadata = context->client_metadata();
        auto fileNameGetter = metadata.find(fileNameKey);
        auto clientIdGetter = metadata.find(clientKey);
        auto mtimeGetter = metadata.find(mtimeKey);

        if (fileNameGetter == metadata.end()){
            dfs_log(LL_ERROR) << "Can't find file";
            stringstream ss;
            ss << "Can't find file" << endl;
            return Status(StatusCode::INTERNAL, ss.str());
        }
        else if (clientIdGetter == metadata.end()){
            dfs_log(LL_ERROR) << "Can't find ClientId";
            stringstream ss;
            ss << "Can't find ClientId" << endl;
            return Status(StatusCode::INTERNAL, ss.str());
        }
        else if (mtimeGetter == metadata.end()){
            dfs_log(LL_ERROR) << "Can't find mTime";
            stringstream ss;
            ss << "Can't find mTime" << endl;
            return Status(StatusCode::INTERNAL, ss.str());
        }

        auto fileName = string(fileNameGetter->second.begin(), fileNameGetter->second.end());
        auto clientId = string(clientIdGetter->second.begin(), clientIdGetter->second.end());
        long mtime = stol(string(mtimeGetter->second.begin(), mtimeGetter->second.end()));

        const string& path = WrapPath(fileName);

        fileNameToClientIdAccess.lock_shared();
        auto lockClientIdGetter = fileNameToClientIdMap.find(fileName);
        string lockClientId;

        if (lockClientIdGetter == fileNameToClientIdMap.end()){
            fileNameToClientIdAccess.unlock_shared();

            stringstream ss;
            ss << "Client Id has no write lock" << fileName;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::INTERNAL, ss.str());

        } else if ((lockClientId = lockClientIdGetter->second).compare(clientId) != 0) {
            fileNameToClientIdAccess.unlock_shared();

            stringstream ss;
            ss << "File is already locked by other client";
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::INTERNAL, ss.str());
        }
        fileNameToClientIdAccess.unlock_shared();

        shared_timed_mutex* fileAccessMutex = getFileAccessMutex(fileName);

        serverMutex.lock();
        fileAccessMutex->lock();

        Status checkSumResult = performChecksum(metadata, path);
        if (!checkSumResult.ok()){
            struct stat status;
            if (checkSumResult.error_code() == StatusCode::ALREADY_EXISTS && stat(path.c_str(), &status) == 0 && mtime > status.st_mtime){
                dfs_log(LL_SYSINFO) << "Need to Update MTime";
                struct utimbuf ub;
                ub.modtime = mtime;
                if (!utime(path.c_str(), &ub)) {
                    dfs_log(LL_SYSINFO) << "Updated MTime";
                }
            }
            releaseMutex(fileName);
            fileAccessMutex->unlock();
            serverMutex.unlock();

            dfs_log(LL_ERROR) << checkSumResult.error_message();
            return checkSumResult;
        }

        dfs_log(LL_SYSINFO) << "Writing: " << path;

        fileData data;
        ofstream writer;
        try {
            while (reader->Read(&data)) {
                if (!writer.is_open()) {
                    writer.open(path, ios::trunc);
                }

                if (context->IsCancelled()){
                    releaseMutex(fileName);
                    fileAccessMutex->unlock();
                    serverMutex.unlock();

                    const string& err = "Deadline Exceeded.";
                    dfs_log(LL_ERROR) << err;
                    return Status(StatusCode::DEADLINE_EXCEEDED, err);
                }

                const string& chunkStr = data.contents();
                writer << chunkStr;
                dfs_log(LL_SYSINFO) << "Wrote: " << chunkStr.length();
            }
            writer.close();
            releaseMutex(fileName);

        } catch (exception const& err) {
            releaseMutex(fileName);
            fileAccessMutex->unlock();
            serverMutex.unlock();

            dfs_log(LL_ERROR) << "Can't write file.";
            stringstream ss;
            ss << "Can't write file." << endl;
            return Status(StatusCode::INTERNAL, ss.str());
        }

        fileStatus status;
        if (getMetadata(path, &status) != 0) {
            fileAccessMutex->unlock();
            serverMutex.unlock();

            stringstream ss;
            ss << "File " << path << " failed get: " << strerror(errno) << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::NOT_FOUND, ss.str());
        }
        fileAccessMutex->unlock();
        serverMutex.unlock();

        response->set_name(fileName);
        Timestamp* modified = new Timestamp(status.modified());
        response->set_allocated_modified(modified);
        return Status::OK;
    }

    Status deleteFile(ServerContext* context, const file* request, fileAck* response) override {

        const string& path = WrapPath(request->name());

        const multimap<string_ref, string_ref>& clientMetadata = context->client_metadata();
        auto clientIdGetter = clientMetadata.find(clientKey);
        if (clientIdGetter == clientMetadata.end()){
            stringstream ss;
            ss << "Missing clientKey in client metadata" << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::INTERNAL, ss.str());
        }
        auto clientId = string(clientIdGetter->second.begin(), clientIdGetter->second.end());
        
        addFileAccessMutex(request->name());
        shared_timed_mutex* fileAccessMutex = getFileAccessMutex(request->name());

        fileNameToClientIdAccess.lock_shared();
        string lockClientId;
        auto lockClientIdGetter = fileNameToClientIdMap.find(request->name());
        if (lockClientIdGetter == fileNameToClientIdMap.end()){
            fileNameToClientIdAccess.unlock_shared();

            stringstream ss;
            ss << "Client Id has no write lock";
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::INTERNAL, ss.str());

        } else if ((lockClientId = lockClientIdGetter->second).compare(clientId) != 0) {
            fileNameToClientIdAccess.unlock_shared();

            stringstream ss;
            ss << "File is already locked by other client";
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::INTERNAL, ss.str());
        }

        fileNameToClientIdAccess.unlock_shared();

        dfs_log(LL_SYSINFO) << "Delete: " << path;

        serverMutex.lock();
        fileAccessMutex->lock();

        fileStatus status;

        if (getMetadata(path, &status) != 0) {
            releaseMutex(request->name());
            serverMutex.unlock();
            fileAccessMutex->unlock();

            stringstream ss;
            ss << "File doesn't exist" << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::NOT_FOUND, ss.str());
        }

        if (context->IsCancelled()){
            releaseMutex(request->name());
            serverMutex.unlock();
            fileAccessMutex->unlock();

            const string& err = "Deadline Exceeded.";
            dfs_log(LL_ERROR) << err;
            return Status(StatusCode::DEADLINE_EXCEEDED, err);
        }

        if (remove(path.c_str()) != 0) {
            releaseMutex(request->name());
            serverMutex.unlock();
            fileAccessMutex->unlock();

            stringstream ss;
            ss << "Removing file failed: " << strerror(errno) << endl;
            return Status(StatusCode::INTERNAL, ss.str());
        }
        
        releaseMutex(request->name());
        serverMutex.unlock();
        fileAccessMutex->unlock();

        response->set_name(request->name());
        Timestamp* modified = new Timestamp(status.modified());
        response->set_allocated_modified(modified);
    
        dfs_log(LL_SYSINFO) << "Succssfully deleted file.";

        return Status::OK;
    }

    Status listFiles(ServerContext* context, const dfs_service::empty* request, files* response) override {
        
        serverMutex.lock_shared();
        DIR *dir;
        if ((dir = opendir(mount_path.c_str())) == NULL) {
            serverMutex.unlock_shared();

            stringstream ss;
            ss << "Failed to Open Directory." << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::NOT_FOUND, ss.str());
        }
        
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {

            struct stat dirStat;
            string dirEntry(entry->d_name);
            string path = WrapPath(dirEntry);
            stat(path.c_str(), &dirStat);
            
            if (!S_ISREG(dirStat.st_mode)){
                dfs_log(LL_SYSINFO) << "Not a file.";
                continue;
            }

            dfs_log(LL_SYSINFO) << "Is a File. ";

            fileStatus* ack = response->add_file();
            fileStatus status;
            if (getMetadata(path, &status) != 0) {
                serverMutex.unlock_shared();

                stringstream ss;
                ss << "Read file failed: " << strerror(errno) << endl;
                dfs_log(LL_ERROR) << ss.str();
                return Status(StatusCode::NOT_FOUND, ss.str());
            }
            ack->set_name(dirEntry);
            Timestamp* modified = new Timestamp(status.modified());
            ack->set_allocated_modified(modified);
        }
        
        serverMutex.unlock_shared();
        closedir(dir);

        return Status::OK;
    }

    Status CallbackList(ServerContext* context, const file* request, files* response) override {
        dfs_service::empty req;
        return this->listFiles(context, &req, response);
    }


    Status getStatus(ServerContext* context, const file* request, fileStatus* response) override {
        
        if (context->IsCancelled()){
            const string& err = "Deadline Exceeded";
            dfs_log(LL_ERROR) << err;
            return Status(StatusCode::DEADLINE_EXCEEDED, err);
        }

        string path = WrapPath(request->name());

        addFileAccessMutex(request->name());
        shared_timed_mutex* fileAccessMutex = getFileAccessMutex(request->name());

        fileAccessMutex->lock_shared();

        if (getMetadata(path, response) != 0) {
            fileAccessMutex->unlock_shared();

            stringstream ss;
            ss << "Read file failed: " << strerror(errno) << endl;
            dfs_log(LL_ERROR) << ss.str();
            return Status(StatusCode::NOT_FOUND, ss.str());
        }

        fileAccessMutex->unlock_shared();

        return Status::OK;
    }

};


DFSServerNode::DFSServerNode(const std::string &server_address,
        const std::string &mount_path,
        std::function<void()> callback) :
        server_address(server_address),
        mount_path(mount_path) {}


DFSServerNode::~DFSServerNode() noexcept {

}

void DFSServerNode::Start() {
    DFSServiceImpl service(this->mount_path, this->server_address);
    service.Run();
}
