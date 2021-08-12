FROM registry.eps.surrey.ac.uk/cuda10skt:13351

ADD ./README.md ./README.md
ADD ./Dockerfile ./Dockerfile
ADD ./utils ./utils
ADD ./models ./models
ADD ./core ./core
ADD ./metrics ./metrics
ADD ./dependencies ./dependencies
ADD ./benchmarks ./benchmarks
ADD ./mturk ./mturk
ADD ./*.py ./
ADD ./git_hash.txt ./
ADD ./prep_data/word_embedding/*.npy ./prep_data/word_embedding/
ADD ./prep_data/sketch_token/* ./prep_data/sketch_token/
