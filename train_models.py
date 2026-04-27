import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import joblib

# Paths
DATASET_DIR = "dataset"
MODELS_DIR = "backend/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Data for DL (VGG16 & Hybrid)
print("--- Loading Data for DL ---")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2. Train VGG16
print("\n--- Training VGG16 ---")
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_vgg.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)
vgg_model = Model(inputs=base_vgg.input, outputs=predictions)

# Freeze base layers
for layer in base_vgg.layers:
    layer.trainable = False

vgg_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
vgg_model.fit(train_generator, validation_data=val_generator, epochs=5)
vgg_model.save(os.path.join(MODELS_DIR, "vgg16_model.h5"))
print("[✓] VGG16 model saved.")

# 3. Train Hybrid (CNN Feature Extractor + SVM)
print("\n--- Training Hybrid (CNN + SVM) ---")
# Use the VGG16 features as a simple hybrid approach (as seen in notebook)
feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-3].output)

def get_features(generator):
    features = []
    labels = []
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        batch_features = feature_extractor.predict(batch_x, verbose=0)
        features.append(batch_features)
        labels.append(np.argmax(batch_y, axis=1))
    return np.vstack(features), np.concatenate(labels)

X_train_dl, y_train_dl = get_features(train_generator)
X_val_dl, y_val_dl = get_features(val_generator)

hybrid_svm = SVC(kernel='linear', probability=True)
hybrid_svm.fit(X_train_dl, y_train_dl)
# The backend expects a .pkl that contains the SVM or a pipeline. 
joblib.dump(hybrid_svm, os.path.join(MODELS_DIR, "hybrid_model.pkl"))
# The guide mentions saving the CNN extractor too
feature_extractor.save(os.path.join(MODELS_DIR, "hybrid_cnn.h5"))
print("[✓] Hybrid model components saved.")

# 4. Load Data for ML (KNN & SVM)
print("\n--- Loading Data for ML (KNN & SVM) ---")
ml_X = []
ml_y = []
class_names = sorted(os.listdir(DATASET_DIR))
class_names = [c for c in class_names if os.path.isdir(os.path.join(DATASET_DIR, c))]

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_DIR, class_name)
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('L').resize((64, 64))
            ml_X.append(np.array(img).flatten())
            ml_y.append(label)

X_ml = np.array(ml_X)
y_ml = np.array(ml_y)

X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_ml = scaler.fit_transform(X_train_ml)
X_test_ml = scaler.transform(X_test_ml)

# 5. Train KNN
print("\n--- Training KNN ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_ml, y_train_ml)
joblib.dump(knn, os.path.join(MODELS_DIR, "knn_model.pkl"))
print("[✓] KNN model saved.")

# 6. Train SVM
print("\n--- Training SVM ---")
svm = SVC(kernel='rbf', C=10, probability=True)
svm.fit(X_train_ml, y_train_ml)
joblib.dump(svm, os.path.join(MODELS_DIR, "svm_model.pkl"))
print("[✓] SVM model saved.")

print("\n--- ALL MODELS TRAINED AND SAVED ---")
