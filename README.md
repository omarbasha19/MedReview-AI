# **MedReview AI – Sentiment Analysis for Pharmacy Customer Feedback**

## **Overview**
MedReview AI is a **sentiment analysis system** designed to classify customer feedback for **Aladdin Pharmacy in Egypt**. The system processes customer reviews and categorizes them into **Positive, Negative, or Neutral** sentiments using **Natural Language Processing (NLP) and Machine Learning** techniques.

### **Why is this project important?**
- Helps the pharmacy **analyze customer satisfaction** and identify trends.
- Provides insights into **which medicines receive frequent negative feedback** for potential improvements.
- Enables pharmacy management to **track product performance** based on customer experiences.
- Generates **interactive reports** to support decision-making.

---

## **Data and Features**
The dataset consists of **1,500 simulated customer reviews**, including information about **medicine names, categories, customer demographics, and sentiment classification**.

### **Dataset Structure**
| Column Name          | Description |
|----------------------|------------|
| **customer_name**   | Simulated names of customers. |
| **gender**         | Gender of the customer (Male/Female). |
| **medicine_name**   | Name of the medicine being reviewed. |
| **type**           | Category of the medicine (e.g., Antibiotic, Pain Relief, Digestive Aid, etc.). |
| **customer_feedback** | Text-based review provided by the customer. |
| **sentiment**      | Sentiment label (Positive, Negative, Neutral). |

---

## **How It Works**
The system follows a structured pipeline for sentiment analysis:

1. **Data Collection & Preprocessing**  
   - Load and clean the dataset.
   - Remove missing values and duplicates.
   - Perform **Tokenization, Stopword removal, Lemmatization**.

2. **Feature Engineering**  
   - Convert text data into **TF-IDF vectors** or **Word Embeddings**.
   
3. **Sentiment Classification Using Machine Learning Models**  
   - Train multiple classification models.
   - Evaluate performance using standard metrics.

4. **Data Visualization**  
   - Generate insights through bar charts, heatmaps, and word clouds.

---

## **Models Used**
The project leverages several **Machine Learning and Deep Learning models** to classify customer sentiment:

### **Baseline Model: Traditional Machine Learning**
- **Logistic Regression**  
- **Random Forest Classifier**  
- **Support Vector Machines (SVM)**  
- **Naive Bayes (MultinomialNB)**  

These models provide a basic benchmark for sentiment classification.

### **Deep Learning Model: Transformer-Based NLP**
- **DistilBERT (Distilled BERT)**  
  - A **lighter version of BERT**, optimized for text classification.
  - Pretrained on large datasets and fine-tuned on pharmacy customer reviews.

- **LSTM (Long Short-Term Memory)**  
  - A **recurrent neural network (RNN)** designed to handle sequential text data.
  - Captures **contextual dependencies** in long reviews.

- **CNN for Text Classification**  
  - A **Convolutional Neural Network (CNN)** model adapted for NLP.
  - Detects important text patterns for better sentiment classification.

### **Performance Metrics Used**
The model's performance is evaluated using:
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix** to understand misclassification rates.

---

## **Visualization and Insights**
### **Sentiment Distribution**
- Most customer reviews are classified as **Positive**, followed by **Neutral**, with fewer **Negative** ones.
- **Pain Relief medicines** receive the highest number of negative reviews.

### **Heatmap Analysis**
- Strong correlation between **medicine type** and customer sentiment.
- Identifies **problematic products** with consistent negative feedback.

### **Key Takeaways**
- **Medicine Name & Category** significantly influence sentiment classification.
- Certain **demographics (e.g., gender differences)** may impact feedback trends.

---

## **Technologies Used**
The project utilizes the following technologies and frameworks:

- **Python** – Core language for NLP and ML.
- **TensorFlow/Keras** – Deep learning for text classification.
- **NLTK & Scikit-learn** – Preprocessing, feature extraction, and traditional ML models.
- **Pandas & NumPy** – Data handling and numerical operations.
- **Matplotlib & Seaborn** – Data visualization.

---

## **Future Enhancements**
- **Expanding Dataset** – Incorporating real-world pharmacy reviews for better accuracy.
- **Improving Model Performance** – Fine-tuning Transformer models like **BERT** and **XLNet**.
- **Integrating AI Chatbots** – Automating responses based on sentiment classification.
- **Multilingual Support** – Expanding sentiment analysis to multiple languages.
- **Market Trend Analysis** – Predicting pharmacy sales trends based on sentiment data.

---

## **Conclusion**
MedReview AI provides a **data-driven approach** to understanding pharmacy customer feedback. By analyzing customer sentiment, the system helps **pharmacy managers, staff, and product teams** make informed decisions to enhance service quality and product offerings.

This project demonstrates the **power of NLP and AI** in real-world business applications, particularly in healthcare and retail sectors.
