# Garbage-Classification
This project focuses on developing a machine learning model to classify garbage into multiple categories. The aim is to automate the sorting process by identifying the type of waste such as plastic, glass, paper, metal, or organic material based on the provided images. The project can serve as a stepping stone toward building more complex waste management systems that contribute to recycling and environmental sustainability.

## Table of Content

- [Read Dataset](#Read_Dataset)
- [Visualization](#Visualization)
- [Modeling](#Modeling)
- [Confusion_matrix](#Confusion_matrix)
- [Accuracy](#Accuracy)

  ## Read_Dataset
  #Create Files_Name
- image_data= '/kaggle/input/garbage-classification/garbage_classification'
- pd.DataFrame(os.listdir(image_data),columns=['Files_Name'])

  ## Visualization
  - sns.countplot(x = dataframe["Label"])
  - plotter.xticks(rotation = 50);
  - ![image](https://github.com/user-attachments/assets/efed8845-e18b-4dfe-a2c8-efa00091f0e9)
 
  ## Modeling
  - base_model = tf.keras.applications.EfficientNetV2B1(input_shape=(224,224,3), include_top=False, weights='imagenet')
  - base_model.trainable = False
  - keras_model = keras.models.Sequential()
  - keras_model.add(base_model)
  - keras_model.add(keras.layers.Flatten()) 
  - keras_model.add(keras.layers.Dropout(0.5))
  - keras_model.add(keras.layers.Dense(12, activation=tf.nn.softmax))     # 12 classes
  - keras_model.summary()
 
  ## Confusion_matrix
  - ax = plt.subplot()
  - CM = confusion_matrix(y_val, y_pred)
  - sns.heatmap(CM, annot=True, fmt='g', ax=ax, cbar=False, cmap='RdBu')
  - ax.set_xlabel('Predicted labels')
  - ax.set_ylabel('True labels') 
  - ax.set_title('Confusion Matrix')
  - plt.show()
  - CM
  - ![image](https://github.com/user-attachments/assets/8504b495-fe38-4c41-b1ac-a443adde5936)

## Accuracy
- Acc = accuracy_score(y_val, y_pred)
- print("accuracy is: {0:.4f}%".format(Acc * 100))
- accuracy is: 99.0654%



