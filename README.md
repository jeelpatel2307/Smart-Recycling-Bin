```markdown
# Smart Recycling Bin

## Overview
The Smart Recycling Bin is a Java application that uses machine learning to classify and sort recyclable materials. By leveraging TensorFlow's Java API, the application can analyze images of waste items and determine whether they belong to categories like plastic, metal, paper, or glass.

## Features
- **Image Classification**: Analyzes images of waste items to classify them into categories such as plastic, metal, paper, or glass.
- **Preprocessing**: Resizes and normalizes input images to the format expected by the machine learning model.
- **Model Inference**: Utilizes a pre-trained TensorFlow model to predict the category of the waste item.

## Requirements
- Java 11 or higher
- Maven
- Pre-trained TensorFlow model (SavedModel format)

## Dependencies
This project uses the TensorFlow Java API. Add the following dependencies to your `pom.xml`:

```xml
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow</artifactId>
    <version>2.6.0</version>
</dependency>
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow-core-platform</artifactId>
    <version>2.6.0</version>
</dependency>
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow-core-api</artifactId>
    <version>2.6.0</version>
</dependency>
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-io</artifactId>
    <version>1.3.2</version>
</dependency>

```

## Setup

1. **Clone the repository**:
    
    ```
    git clone <https://github.com/yourusername/smart-recycling-bin.git>
    cd smart-recycling-bin
    
    ```
    
2. **Build the project using Maven**:
    
    ```
    mvn clean install
    
    ```
    
3. **Download the pre-trained TensorFlow model**:
    - Ensure you have a pre-trained TensorFlow model saved in the SavedModel format.
    - Place the model in an accessible directory (e.g., `path/to/saved_model`).
4. **Run the application**:
    
    ```
    mvn exec:java -Dexec.mainClass="SmartRecyclingBin"
    
    ```
    

## Usage

To use the Smart Recycling Bin, run the application and provide the path to an image of a waste item. The application will classify the image and output the predicted category.

### Example

1. **Input Image**: Provide an image file (e.g., `path/to/image.jpg`).
2. **Output**:
    
    ```
    Predicted class: Plastic
    
    ```
    

## Code Overview

### Main Class

The main class, `SmartRecyclingBin.java`, performs the following tasks:

- Loads the TensorFlow model.
- Preprocesses the input image (resizes and normalizes).
- Runs the model inference to classify the image.
- Outputs the predicted category.

```java
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;

public class SmartRecyclingBin {

    public static void main(String[] args) {
        // Load the TensorFlow model
        try (SavedModelBundle model = SavedModelBundle.load("path/to/saved_model")) {
            // Load and preprocess the image
            BufferedImage image = ImageIO.read(new File("path/to/image.jpg"));
            Tensor<TFloat32> inputTensor = preprocessImage(image);

            // Perform the inference
            Tensor<TFloat32> outputTensor = model.session()
                    .runner()
                    .feed("input_tensor", inputTensor)
                    .fetch("output_tensor")
                    .run()
                    .get(0)
                    .expect(TFloat32.DTYPE);

            // Process the output
            float[] probabilities = outputTensor.data().asRawTensor().data().asFloats().toArray();
            int maxIndex = getMaxIndex(probabilities);

            // Print the result
            String[] labels = {"Plastic", "Metal", "Paper", "Glass"};
            System.out.println("Predicted class: " + labels[maxIndex]);

        } catch (IOException e) {
            System.err.println("Failed to read image: " + e.getMessage());
        }
    }

    private static Tensor<TFloat32> preprocessImage(BufferedImage image) {
        int width = 224;  // Example input size expected by the model
        int height = 224;

        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        resizedImage.getGraphics().drawImage(image, 0, 0, width, height, null);

        float[] imageData = new float[width * height * 3];
        int[] pixels = resizedImage.getRGB(0, 0, width, height, null, 0, width);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            imageData[i * 3] = ((pixel >> 16) & 0xFF) / 255.0f;  // Red
            imageData[i * 3 + 1] = ((pixel >> 8) & 0xFF) / 255.0f;  // Green
            imageData[i * 3 + 2] = (pixel & 0xFF) / 255.0f;  // Blue
        }

        return TFloat32.tensorOf(Shape.of(1, width, height, 3), FloatBuffer.wrap(imageData));
    }

    private static int getMaxIndex(float[] probabilities) {
        int maxIndex = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}

```
