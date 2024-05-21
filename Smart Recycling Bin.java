<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>smart-recycling-bin</artifactId>
    <version>1.0-SNAPSHOT</version>

    <dependencies>
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
    </dependencies>
</project>










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
