# SWSS
The Smart Waste Sorting System (SWSS) is an automated solution that uses image recognition and motorized mechanisms to identify and sort waste items efficiently. This repository contains the source code developed for this engineering project, serving as an archive of the work.
<br>
<br>
This Smart Waste Sorting System was a 1st place winner in the 2025 First Year Engineering Design Day (FEDD) at North Carolina State University.
<br>
<br>
### Credits to Contributors:
<ul>
    <li>Keith Hendricks (kdhendr2@ncsu.edu)</li>
    <li>Skyler Looney (saloone2@ncsu.edu)</li>
    <li>tjlewis4@ncsu.edu (tjlewis4@ncsu.edu)</li>
    <li>Brandon Mauldin (bsmauldi@ncsu.edu)</li>
</ul>

# Presentation Poster
![Presentation Poster](./docs/E101%20Presentation.png)

# Simplified UML Class Diagram
![UML Class Diagram](./docs/Simple%20UML%20Class%20Diagram.png)

# Photos and Videos
![SWSS in the Early Stages](./docs/SWSS%20Early%20Stages.png)
[![Video Title](https://img.youtube.com/vi/zS_YD7wb0NA/0.jpg)](https://youtu.be/zS_YD7wb0NA)

# Requirements

### Requirements.txt
Contains all of the necessary python libraries/modules that should be installed for the project.

### Underlying Hardware Components in Designed Frame
<b>Note:</b> The following components are hardware that is used in the underlying frame but NOT an exhaustive list
of all materials like (wood, nails, etc.)

* Raspberry Pi 4 Model B
* Raspberry Pi Camera Module 2
* 28BYJ-48 Stepper Motor & ULN2003 drivers
* Breadboard jumper wires (Female to Female)
 
### Model Dataset
The WasteNet.keras file is a serialized representation of the WasteNet neural network model, encapsulating its architecture, weights, and configuration. This file must be generated before executing the main program.
<br>
<br>
To create the WasteNet.keras file, download and unzip the dataset, then run the WasteNet.create_model method. On its first execution, this method will use the dataset to build and save the model as WasteNet.keras for subsequent use.
<br>
<br>
<b>Dataset:</b> https://kaggle.com/datasets/57b0b902dba6a467c6ec0354b52616d11835986d72a7104a46632b9330d9008d
