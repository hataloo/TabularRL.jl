docker build -t tabularrl.jl .
docker run --rm -it -v "$(pwd)":/TabularRL.jl tabularrl.jl 