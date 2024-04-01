# Master_Project_Mohammed_ALBHAISI

## Description

The title of my master project was “Improvement of object detection candidate certainty using redundant tracklets and situational information”. The main goal of this project is to increase the certainty of detection approaches, to increase the safety of
autonomous vehicles. Different Tools are used, such as nuscenes devkit, visual studio code, python, and latex

Different object detection approaches are used to detect different classes. All detected objects will be tracked using the Immortal Tracker algorithm. The idea of joint detection and tracking is to increase safety and precision. This project focuses on object detection and tracking and how different approaches can work together, aiding each other in order to improve performance. To make a general overview of all stages that are followed in this project:

1- At the first stage, the detection results will be obtained using different object detection approaches.
2- At the second stage, all detected objects will be tracked using Immortal Tracker.
3- At the third stage, all detection and tracking results will be evaluated using different
metrics.
4- At the final stage, based on evaluation results and which result is better than another, the detection approaches results can be improved.

Remark: The master-thesis will be uploaded here

## Basics

### 1 Submodules

Add other repositories as submodules using:
(!!! If and only if no changes are required or changes can be commitet !!!)
```
git submodule add <remote_url> <destination_folder>
```

Make sure that all submodules are initialized and updated:
```
git submodule update --init --recursive
```
### 2 Adapt path for volumes

Go to docker-commpose.yml and adapt the paths used for the volumes according to your system.

DO NOT COMMIT THIS CHANGES.

### 3 build the dev container

Build dev container using docker-compose:
```
sudo docker-compose up --build
```
### Connect vs code to the container

following extensions are recommended / necesarry:
vsc-extensions.txt holds extensions that are recommended.
To install them execute the following commands in vsc while in the project folder:
```
ctrl + P

ext install aslamanver.vsc-export

ctrl + shift + P

VSC Import
```

You should be able to see the container via your remote explorer.


Click the folder icon to open your workspace in the container.


# SRC

## test

Run simple code from nuScenes devkit tutorial to test nuscenes.
Tested.

## Matching-threshold

Definition for different matching threshold using IOU and GIOU


# Detection-Tracking metrics results

## Tracking metric results

### Tracking evaluation metric for PointPillar original 

Work with tracking evaluation code for PointPillar original from nuscenes
download the json file from 
https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any

### Render different Ground-Prediction boxes 

Render some GT and Pd as well Ids for different scenes for class car

![1537295833898809](Detection-Tracking-metrics-result/Documentation/centerpoint_02pillar-LOOCV01/3ada261efee347cba2e7557794f1aec8/car/1537295833898809.png)

### Visualize one instance 

![Tracking-car](Detection-Tracking-metrics-result/Documentation/Tracking-car.jpg)

## Detection evaluation metrics result

include the final metrics results for different detection architectures 
(PointPillar, CenterPoint Pillar02, Voxel01, Voxel0075)

### Visualize detection for classes 

Render example for centerpoint-pillar02 

![detection-centerpoint-pillar02](Detection-Tracking-metrics-result/Documentation/detection-centerpoint-pillar02.png)


### Visualize precision-recall curve for class car

 ![car-pr](Detection-Tracking-metrics-result/Documentation/car-pr.png)












































## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.uni-due.de/srs/students/ma_rajput.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.uni-due.de/srs/students/ma_rajput/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
