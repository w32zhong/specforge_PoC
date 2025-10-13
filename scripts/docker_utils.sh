CR_PREFIX=${1-ghcr.io/w32zhong}

function build() {
    module=$1
    shift 1
    set -x
    docker build -f ${module}_dockerfile --build-arg MODEL=$module -t $module $@ .
    docker image tag $module ${CR_PREFIX}/${module}:latest
    set +x
}

function push() {
    module=$1
    set -x
    docker push ${CR_PREFIX}/${module}:latest
    set +x
}

function build_and_push() {
    module=$1
    build $module
    push $module
}

function cleanup() {
    docker container prune -f
    docker image prune -f
    docker images
}
