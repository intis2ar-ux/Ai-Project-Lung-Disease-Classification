Here's a detailed **README.md** file for your project, which you can directly use or adapt:

---

# **AI Project: Lung Disease Classification**

### **Overview**
This project focuses on classifying lung diseases (Normal vs. Pneumonia) using chest X-rays. It leverages convolutional neural networks (CNNs) to achieve high accuracy and confidence in predictions. The models used are ResNet50, DenseNet121, and MobileNetV3, trained on the NIH Chest X-Ray dataset.

---

### **Project Objectives**
1. Prepare and preprocess the NIH Chest X-Ray dataset.
2. Train and evaluate three CNN models:
   - **ResNet50**
   - **DenseNet121**
   - **MobileNetV3**
3. Compare the models based on:
   - Accuracy
   - Mean Average Precision (mAP)
   - Training Time
4. Recommend the best-performing model.

---

### **Dataset**
The project uses the **NIH Chest X-Ray Dataset**, which contains labeled chest X-rays for conditions like "No Finding" (Normal) and "Pneumonia".

- **Data Preparation Steps**:
  1. Filtered the dataset for "Normal" and "Pneumonia" classes.
  2. Organized images into labeled folders.
  3. Split the dataset into **training (70%)**, **validation (20%)**, and **testing (10%)** sets.
  4. Resized all images to `(224, 224)` to match the input size of the CNN models.

---

### **Modeling**
The following CNN architectures were used:
1. **ResNet50**: A deep residual network with 50 layers.
2. **DenseNet121**: A densely connected convolutional network.
3. **MobileNetV3**: A lightweight network optimized for mobile and edge devices.

#### **Training Configuration**
- Optimizer: **Adam** with a learning rate of `0.0001`.
- Loss Function: **Categorical Crossentropy**.
- Epochs: **50** (Early stopping applied).
- Batch Size: **32**.

#### **Evaluation Metrics**
1. **Accuracy**: Classification correctness.
2. **Mean Average Precision (mAP)**: Confidence in predictions.

---

### **Results**

| **Model**       | **Test Accuracy** | **mAP**  | **Training Time (s)** | **Epochs Trained** |
|------------------|-------------------|----------|-----------------------|--------------------|
| **ResNet50**     | 97.88%           | 0.4987   | 41                   | 8                  |
| **DenseNet121**  | 98.29%           | 0.7346   | 46                   | 32                 |
| **MobileNetV3**  | 97.88%           | 0.5018   | 18                   | 9                  |

#### **Conclusion**
- **Best Model**: **DenseNet121**
  - Achieved the highest accuracy (**98.29%**) and mAP (**0.7346**), making it the most reliable model for lung disease classification.
- **Efficient Alternative**: **MobileNetV3**
  - Demonstrated comparable accuracy (**97.88%**) with the shortest training time (**18 seconds**), suitable for resource-constrained environments.

---

### **Dependencies**
The following libraries and tools were used in the project:
- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

### **Usage Instructions**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Ai-Project-Lung-Disease-Classification.git
   cd Ai-Project-Lung-Disease-Classification
   ```

2. **Prepare the Dataset**:
   - Download the NIH Chest X-Ray Dataset from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC).
   - Place the dataset in the `data/` folder and preprocess using the provided code.

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook LungProjAi.ipynb
   ```

4. **Test Images**:
   - Use the `test_single_image` or `test_and_display_images` functions to validate the models on individual or multiple images.

---

### **References**
1. **NIH Chest X-Ray Dataset**:  
   [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
2. **Keras Applications Documentation**:  
   [https://keras.io/api/applications/](https://keras.io/api/applications/)
3. **Tutorial on Transfer Learning**:  
   [https://www.youtube.com/watch?v=4kN7H55a5u4](https://www.youtube.com/watch?v=4kN7H55a5u4)
4. **mAP Metric Explanation**:  
   [https://www.analyticsvidhya.com/blog/2021/04/understanding-mean-average-precision-map-for-object-detection/](https://www.analyticsvidhya.com/blog/2021/04/understanding-mean-average-precision-map-for-object-detection/)
5. **DenseNet Architecture Paper**:  
   [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)

