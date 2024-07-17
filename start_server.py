import sys
from ParallelCollaborativeInference import ParallelCollaborativeInference

if __name__ == '__main__':
    ip = "192.168.50.11"
    port = 12345
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    PCI = ParallelCollaborativeInference(parallel_approach="select")
    PCI.start_server()
