# bumblebee
## Setup gitlab
Go to https://gitlab.eps.surrey.ac.uk/ create an empty repo and name it e.g. lstm. My git repo
then has a link: https://gitlab.eps.surrey.ac.uk/tb00083/lstm
on the same gitlab repo "Settings/General/Visibility, project features, permissions" enable "Container registry"

## Setup docker
on manfred, clone the git repo which u want to build docker image upon, cd to that folder.
You should see the Dockerfile in that folder.
Edit the Dockerfile and remove any line that setup jupyter notebook; then add this line to the end:

EXPOSE 5000

Next, build the docker image and name it e.g. lstm:base

$docker login gitlab-registry.eps.surrey.ac.uk
$docker build --rm -t registry.eps.surrey.ac.uk/tb00083/lstm:base  . 
$docker push registry.eps.surrey.ac.uk/tb00083/lstm:base

You have successfully register a docker image named lstm:base in gitlab. You can now setup condor docker image file 
as registry.eps.surrey.ac.uk/tb00083/lstm:base.



