from dotenv import load_dotenv
import os
import sys
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import Azure namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Load configuration from .env
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        if not ai_endpoint or not ai_key:
            raise Exception("Please set AI_SERVICE_ENDPOINT and AI_SERVICE_KEY in your .env file.")

        # Get image file from argument or default
        image_file = 'images/Lincoln.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Read text in image
        with open(image_file, "rb") as f:
            image_data = f.read()

        print(f"\nReading text in {image_file}...\n")

        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

        # Print lines of text
        if result.read is not None:
            print("Lines of text:")
            for line in result.read.blocks[0].lines:
                print(f" {line.text}")
            annotate_lines(image_file, result.read)

            # Print individual words
            print("\nIndividual words with confidence:")
            for line in result.read.blocks[0].lines:
                for word in line.words:
                    print(f"  {word.text} (Confidence: {word.confidence:.2f}%)")
            annotate_words(image_file, result.read)
        else:
            print("No text detected.")

    except Exception as ex:
        print("Error:", ex)


def annotate_lines(image_file, detected_text):
    print(f'\nAnnotating lines of text in image...')
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width / 100, image.height / 100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for line in detected_text.blocks[0].lines:
        r = line.bounding_polygon
        rectangle = [(r[0].x, r[0].y), (r[1].x, r[1].y),
                     (r[2].x, r[2].y), (r[3].x, r[3].y)]
        draw.polygon(rectangle, outline=color, width=3)

    plt.imshow(image)
    plt.tight_layout(pad=0)
    fig.savefig('lines.jpg')
    print('  Results saved in lines.jpg')


def annotate_words(image_file, detected_text):
    print(f'\nAnnotating individual words in image...')
    image = Image.open(image_file)
    fig = plt.figure(figsize=(image.width / 100, image.height / 100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for line in detected_text.blocks[0].lines:
        for word in line.words:
            r = word.bounding_polygon
            rectangle = [(r[0].x, r[0].y), (r[1].x, r[1].y),
                         (r[2].x, r[2].y), (r[3].x, r[3].y)]
            draw.polygon(rectangle, outline=color, width=3)

    plt.imshow(image)
    plt.tight_layout(pad=0)
    fig.savefig('words.jpg')
    print('  Results saved in words.jpg')


if __name__ == "__main__":
    main()
