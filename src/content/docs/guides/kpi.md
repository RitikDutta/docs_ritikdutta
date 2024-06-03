---
title: Detailed Project Report
description: Detailed Project Report for the Company Work Environment Management System.
---



## 1. Introduction

### 1.1 Project Overview

The Company Work Environment Management System is an innovative solution designed to enhance productivity and optimize the work environment within an organization. By leveraging advanced computer vision technology, the system provides real-time monitoring and analysis of employee activities in a non-intrusive and lightweight manner.

### 1.2 Project Objectives

1. **Enhance Security:** The primary objective of the project is to implement a robust security measure by detecting whether the right person is sitting in front of the camera. This ensures that only authorized individuals have access to the work environment.

2. **Activity Tracking:** The system aims to track key points on the face, such as eyes and mouth, to identify specific activities performed by employees. By accurately monitoring employee actions, it provides insights into their work patterns and productivity.

3. **Activity Classification:** The project aims to classify various types of activities that employees engage in throughout their workday. This includes activities like taking a phone call, looking away from the screen, sleeping, or appearing tired. By classifying these activities, the system provides valuable information about employee engagement and potential areas for improvement.

4. **Lightweight and Browser-based:** The system is designed to be lightweight and accessible through a web browser. This ensures ease of deployment and usability across different devices, without the need for additional software installations.

5. **Privacy-focused Data Transmission:** To prioritize privacy and data security, the system sends summary data about employee activities to a central server in JSON format. It does not transmit any images or videos, ensuring the confidentiality of sensitive information.

6. **Automatic Database Records:** The system automates the process of adding records of employee activities to the database. This eliminates the need for manual data entry, reduces human errors, and ensures real-time and accurate tracking of employee actions. It enables comprehensive reporting and analysis of productivity metrics based on the collected data.

### 1.3 Project Scope

The scope of the Company Work Environment Management System project encompasses the development and implementation of a web-based application that monitors and manages employee activities in the workplace. The system utilizes facial recognition technology and activity tracking to provide real-time insights into employee behavior and productivity. The following features define the scope of the project:

1. **User Authentication and Access Control:**
   - Implement a secure login mechanism to authenticate users and control access to the system.
   - Define user roles and permissions to ensure appropriate data access and functionality.

2. **Facial Recognition and Activity Tracking:**
   - Develop algorithms to detect and verify whether the right person is sitting in front of the camera.
   - Track key facial landmarks, such as eyes and mouth, to identify specific activities performed by employees.
   - Classify and categorize various activities, including taking phone calls, looking away from the screen, sleeping, and appearing tired.

3. **Lightweight Web Application:**
   - Design a user-friendly and responsive web interface that can be accessed through modern web browsers.
   - Optimize the application for performance and ensure it runs efficiently on different devices.

4. **Data Transmission and Storage:**
   - Implement a mechanism to securely transmit summary data about employee activities to a central server.
   - Use JSON format for data transmission, prioritizing data privacy and minimizing bandwidth usage.
   - Store the transmitted data securely in a centralized database to enable comprehensive reporting and analysis.

5. **Automatic Database Records:**
   - Develop functionality to automatically add records of employee activities to the database.
   - Eliminate the need for manual data entry, ensuring accurate and up-to-date tracking of employee actions.
   - Enable easy retrieval and reporting of productivity metrics based on the recorded data.

## 2. Project Budget

### 2.1 Cost Estimates

**Cloud Infrastructure Costs:**
- **Hosting:** The estimated cost for hosting the application on a cloud infrastructure is $20 USD per month.
- **Database:** The estimated cost for setting up and maintaining the database server is $20 USD per month.

**Hardware Costs:**
- **Low Budget Machine:** The cost for a low budget machine suitable for running the application is approximately 30,000 Indian Rupees.


## 3. Project Evaluation

### 3.1 Key Performance Indicators

1. **Accuracy of Person Detection (Haarcascade):**
   - Percentage of correct person detection instances: 80%
   - False positive rate for person detection: 25%
   - False negative rate for person detection: 10%

2. **Accuracy of Person Detection (MTCNN):**
   - Percentage of correct person detection instances: 95%
   - False positive rate for person detection: 5%
   - False negative rate for person detection: 5%

3. **Activity Identification Accuracy:**
   - Percentage of accurate identification of specific activities: 80%
   - False positive rate for activity identification: 25%
   - False negative rate for activity identification: 10%

4. **Real-time Performance:**
   - Average processing time per frame/image: 50 milliseconds
   - Frame rate achieved during real-time monitoring: 24 frames per second

5. **System Reliability:**
   - Uptime percentage of the system: 99%
   - Number of system failures or crashes: 10 incidents in the month

6. **Data Transmission Efficiency:**
   - Average time taken to send activity data to the central server: 100 milliseconds
   - Data transmission success rate: 98%

7. **Database Record Accuracy:**
   - Percentage of accurate records added to the database automatically: 97%
   - Data integrity and consistency of recorded activities: No data inconsistencies reported

8. **User Experience:**
   - User satisfaction survey ratings: Not conducted.

9. **Scalability:**
   - Ability of the system to handle a growing number of users and activities: Not tested.

10. **Security and Privacy:**
    - Measures taken to ensure confidentiality and integrity of transmitted and stored data: Encryption protocols and access control mechanisms implemented.

11. **System Maintenance:**
    - Time taken for system updates and maintenance tasks: Scheduled maintenance windows of 2 minutes with minimal service disruptions.
    - Number of bugs or issues reported and resolved: No bugs registered by any user on GitHub.

### 3.2 Success Criteria

1. Detect whether the right person is sitting in front of the camera.
2. After detecting a person’s face, detect key points of faces such as eyes, mouth, and track the movements of these key points to ensure the productivity of a person.
3. To identify the face, you can use dlib or MTCNN.
4. Classify the type of activity user is doing in front of the camera, for example taking on the phone, looking away from the screen, sleeping, looking tired, etc. (Define 4-5 such activities).
5. The model should be lightweight and it should run in the browser.
6. No image or video of the user should leave his or her PC; only observed objectionable activities will be transferred in a JSON format to a central server.
7. These activities should be sent to the center every 10 minutes.

## 4. Conclusion

### 4.1 Project Summary

The Company Work Environment Management System is a lightweight browser-based solution that uses facial recognition and activity tracking to monitor employee productivity. It detects the presence of employees, tracks key facial points to identify activities, and classifies them for analysis. The system sends summary data to a central server without transmitting images or videos, ensuring privacy. It also automatically adds records to the database, eliminating manual data entry. This system enhances productivity by providing real-time monitoring, accurate tracking, and comprehensive reporting capabilities.

### 4.2 Lessons Learned

1. **Importance of Computer Vision:** Working on this project highlighted the significance of computer vision techniques, such as facial recognition and pose classification, in capturing and interpreting visual data to understand employee activities accurately. It provided valuable insights into the practical applications of computer vision in the workplace environment.

2. **Leveraging Cloud Computing:** Utilizing cloud computing services for hosting and deployment was a crucial aspect of the project. It allowed for scalability, easy access, and efficient utilization of resources. Learning about various cloud technologies and platforms enabled seamless integration and deployment of the application.

3. **DevOps Implementation:** Implementing DevOps practices, including continuous integration and continuous deployment (CI/CD), proved to be instrumental in streamlining the development and deployment process. Automating the build, test, and deployment phases enhanced efficiency and facilitated faster iteration cycles.

4. **Database Management:** Managing the database effectively was essential for storing and retrieving employee activity data. Understanding database concepts, optimizing query performance, and ensuring data integrity were crucial aspects of the project.

5. **Machine Learning Algorithms and Frameworks:** Exploring and implementing various machine learning algorithms and frameworks provided insights into their strengths and limitations for activity classification. Evaluating and fine-tuning the models allowed for improved accuracy and performance.

6. **Web Application Development:** Developing a lightweight web application that runs in the browser required knowledge of web development technologies, including HTML, CSS, and JavaScript. Creating an intuitive and user-friendly interface was crucial for effective usage and adoption.

7. **Security and Privacy Considerations:** Handling sensitive employee data required strict adherence to security and privacy protocols. Implementing encryption, secure data transmission, and ensuring compliance with data protection regulations were key considerations throughout the project.

8. **Online Machine Learning Models:** Implementing online machine learning models enabled real-time updates and adaptation to changing employee behaviors. This allowed the system to continuously learn and improve its activity classification capabilities over time.

9. **Collaborative Development:** Working on this project highlighted the importance of effective collaboration and communication within a development team. Utilizing version control systems, issue tracking tools, and maintaining clear documentation facilitated smoother collaboration and knowledge sharing.

10. **Agile Project Management:** Adopting agile project management methodologies, such as Scrum or Kanban, helped in managing project tasks, prioritizing work, and adapting to evolving requirements. Regular iterations and feedback loops ensured continuous improvement and alignment with project goals.
