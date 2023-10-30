# Quality Assurance within the Automobile Industry
In the automobile industry, steel is a fundamental material used for manufacturing various components, such as chassis, body panels, and structural parts. Ensuring the quality and integrity of steel products is critical to guaranteeing the safety, reliability, and performance of vehicles. However, defects in steel products can occur during the manufacturing process, and detecting these defects is challenging with traditional methods. This problem aims to leverage Artificial Intelligence (AI) and Machine Learning (ML) techniques to detect and predict defects in steel products used in the automobile industry, ultimately enhancing the quality assurance processes.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Methodology](#Methodology)
- [Usage](#usage)
- [Contributing](#contributing)

## About
For the automobile industry's quality control of finished goods, automatic detection of steel surface flaws is crucial. However, due to its poor accuracy and sluggish running speed, the traditional method cannot be effectively used in a production line. There is still much room for improvement with the current, widely used algorithm (based on deep learning), which has the issue of low accuracy. This study suggests a method for combining enhanced faster region convolutional neural networks (faster R-CNN) with improved ResNet50 to decrease average running time and increase accuracy. First, the image is input into the enhanced ResNet50 model, which includes the improved cutout and deformable revolution network (DCN) to classify the sample as having defects or not. The algorithm outputs the sample without defects directly if the probability of a defect is less than 0.3. If not, the samples are further input into the enhanced faster R-CNN, which includes matrix NMS, enhanced feature pyramid networks, and spatial pyramid pooling. The final product includes information about where and what kind of defect there is in the sample, if any. The accuracy of this method can be increased to 98.2 percent by analyzing the data set that was collected in a real-world manufacturing environment. The average running time is quicker than that of other models at the same time.

## Features

List the key features of this project:
- Feature 1: Defect Detection: Develop AI and ML models that can accurately detect various types of defects (e.g., cracks, inclusions, surface abnormalities) in steel products used for automobile manufacturing.
- Feature 2: Defect Classification: Create a system that categorizes detected defects into different classes based on severity and type to help prioritize quality control efforts.
- Feature 3: Predictive Maintenance: Utilize AI and ML to predict potential defects before they occur, enabling proactive maintenance and minimizing production downtime and cost.

## Installation
- !pip install flask
- !pip install pandas
- !pip install numpy
- !pip install tensorflow  

- Use this tutorial for tensorflow  installation https://www.tutorialspoint.com/tensorflow/tensorflow_installation.htm

## Methodology:
- **1.Problem Understanding and Definition**: 
a)	Clearly define the types of defects you want to detect in steel products (e.g., surface defects, cracks, irregularities).
b)	Understand the criticality of these defects in relation to the automobile industry's safety and performance standards.
- **2. Data Collection and Preparation**:
a) Gather a diverse dataset of steel product images or data related to defects. This dataset should encompass various types, severities, and instances of defects.
b) Annotate the data, marking the defects and non-defective areas for supervised learning.
c) Clean and preprocess the data to ensure consistency and relevance.
- **3. Feature Engineering**:
a) Extract relevant features from the data that are crucial for defect detection (e.g., color, texture, shape).
b) Use domain expertise to identify and select the most informative features.
- **4. Model Selection and Training**:
a) Choose appropriate AI/ML models for the defect detection task (e.g., Convolutional Neural Networks - CNN for image data).
b) Split the dataset into training and validation sets.
c) Train the selected models using the training set, fine-tuning hyperparameters as needed.
d) Validate the models using the validation set and iterate on the training process for improved performance.
- **5. Model Evaluation**:
a) Evaluate the models using separate test data to assess their performance and generalization to unseen samples.
b) Metrics like precision, recall, F1 score, and accuracy are typically used for evaluation.
- **6. Model Optimization and Fine-Tuning**:
a) Optimize the models for better performance, considering false positives/negatives and adjusting thresholds for decision-making.
b) Fine-tune the models based on feedback and analysis of model performance.
- **7. Integration into Manufacturing Process**:
a) Develop an interface to integrate the AI/ML model into the manufacturing process for real-time defect detection.
b) Ensure seamless data flow from sensors or inspection points to the AI model and decision-making systems.
- **8. Continuous Monitoring and Improvement**:
a) Continuously monitor the performance of the deployed AI/ML model in the production line.
b) Gather feedback and update the model to adapt to new defect patterns or variations in the manufacturing process.
- **9. Documentation and Training**:
a) Document the entire methodology, including data preprocessing, model selection, and integration processes.
b) Provide training to relevant stakeholders on utilizing and maintaining the AI/ML system.
By following this methodology, you can systematically implement AI and ML solutions to detect defects in steel products used in the automobile industry, enhancing product quality and safety.

## Acknowledgments
- **Open Source AI/ML Community**: I express my gratitude to the open-source AI and ML community for their continuous development of frameworks, libraries, and tools. These resources played a vital role in our project's success.

- **Machine Learning Researchers**: I would like to acknowledge the researchers and scientists in the field of machine learning and artificial intelligence who have contributed to the advancements in AI technology. Their work has been fundamental to my project.
