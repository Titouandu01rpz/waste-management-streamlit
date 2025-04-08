import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Model Definition
class ImageClassificationBase(nn.Module):
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    
        loss = nn.functional.cross_entropy(out, labels)   
        acc = (out.argmax(dim=1) == labels).float().mean() 
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

class ResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

@st.cache_resource
def load_model():
    model = ResNet(num_classes=7)
    model.load_state_dict(torch.load("resnet50_classifier.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Mapping of classes to bins
class_to_bin = {
    'battery': 'Black Bin (Hazardous Waste)',
    'cardboard': 'Yellow Bin (Recyclables)',
    'glass': 'Yellow Bin (Recyclables)',
    'metal': 'Yellow Bin (Recyclables)',
    'organic': 'Black Bin (General Waste)',
    'paper': 'Yellow Bin (Recyclables)',
    'plastic': 'Yellow Bin (Recyclables)'
}

class_names = ['battery', 'cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']

# Embedding CSS Styles (EXACTLY as in your original)
def embed_css():
    css = """
    <style>
    /* Base styles */
    :root {
        --primary: #4CAF50;
        --primary-light: #81C784;
        --primary-dark: #388E3C;
        --secondary: #2196F3;
        --accent: #FFC107;
        --background: #F5F8F5;
        --foreground: #213321;
        --card: #FFFFFF;
        --muted: #F0F4F0;
        --muted-foreground: #637563;
        --border: #E0E8E0;
        /* Bin colors */
        --yellow-bin: #FFEB3B;
        --black-bin: #212121;
    }
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: var(--background);
        color: var(--foreground);
        line-height: 1.5;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234caf50' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2V6h4V4h-4zM0 34v-4H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    .container {
        max-width: 2550px;
        margin: 0 auto;
    }
    header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    header h1 {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(to right, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: var(--muted-foreground);
        font-size: 1rem;
    }
    .icon-large {
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 150px;
        color: var(--primary);
        opacity: 0.1;
        z-index: -1;
    }
    /* Card styles */
    .card {
        background-color: var(--card);
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        overflow: hidden;
        margin-bottom: 1.5rem;
        position: relative;
        border: 2px solid rgba(76, 175, 80, 0.2);
        padding: 1.5rem;
        min-height: 200px;
    }
    .card-header {
        background-color: rgba(76, 175, 80, 0.05);
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-header h2 {
        font-size: 1.25rem;
        font-weight: 600;
    }
    .card-content {
        padding: 1rem;
    }
    .icon-circle {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(76, 175, 80, 0.1);
        width: 32px;
        height: 32px;
        border-radius: 50%;
    }
    .icon-circle i {
        color: var(--primary);
        font-size: 1rem;
    }
    /* Detection results */
    .detection-result {
        margin-top: 1rem;
    }
    .detection-item {
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .detection-header {
        padding: 0.5rem 1rem;
        color: #000;
        font-weight: 600;
    }
    .detection-header.yellow {
        background-color: var(--yellow-bin);
    }
    .detection-header.black {
        background-color: var(--black-bin);
        color: white;
    }
    .detection-info {
        background-color: white;
        padding: 1rem;
        text-align: center;
        font-weight: 500;
    }
    /* Alert card */
    .alert-card {
        background: linear-gradient(to right, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1));
        border: 1px solid rgba(76, 175, 80, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
    }
    .alert-icon {
        color: var(--primary);
        margin-right: 0.5rem;
        display: inline-block;
    }
    .alert-content h3 {
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    .alert-content p {
        margin-bottom: 1rem;
    }
    .bin-categories {
        display: grid;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .bin-category {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.875rem;
        flex-wrap: wrap;
    }
    .color-dot {
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 50%;
    }
    .yellow {
        background-color: var(--yellow-bin);
    }
    .black {
        background-color: var(--black-bin);
    }
    .bin-name {
        font-weight: 600;
        margin-right: 0.25rem;
    }
    .bin-desc {
        color: var(--muted-foreground);
    }
    .recycling-message {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding-top: 0.5rem;
        border-top: 1px solid var(--border);
        color: var(--muted-foreground);
        font-size: 0.75rem;
        font-style: italic;
    }
    .recycling-message i {
        color: var(--primary);
    }
    /* Recycling Bins */
    .bin-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
    }
    .recycling-bin {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        width: 120px;
        min-height: 200px;
    }
    .bin {
        width: 80px;
        height: 100px;
        border-radius: 8px 8px 0 0;
        position: relative;
        margin-top: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .bin:hover {
        transform: translateY(-5px);
    }
    .bin-lid {
        position: absolute;
        top: -15px;
        width: 86px;
        height: 15px;
        border-radius: 5px 5px 0 0;
        background-color: inherit;
        box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        transform-origin: center top;
    }
    .bin:hover .bin-lid {
        transform: rotateX(30deg);
    }
    .bin-label {
        position: absolute;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .yellow-bin {
        background-color: var(--yellow-bin);
    }
    .yellow-bin .bin-lid {
        background-color: #E6D335;
    }
    .black-bin {
        background-color: var(--black-bin);
    }
    .black-bin .bin-lid {
        background-color: #111111;
    }
    .black-bin .bin-label {
        color: white;
    }
    .bin-info {
        margin-top: 0.75rem;
    }
    .bin-title {
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 0.25rem;
    }
    .bin-description {
        font-size: 0.75rem;
        color: var(--muted-foreground);
    }
    /* Sorting information */
    .sorting-info h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .sorting-info p {
        font-size: 0.875rem;
        color: var(--muted-foreground);
        margin-bottom: 1rem;
    }
    .sorting-info ul {
        list-style-position: inside;
        font-size: 0.875rem;
        color: var(--muted-foreground);
    }
    .sorting-info li {
        margin-bottom: 0.5rem;
    }
    /* Animations */
    .recycling-icon {
        animation: spin 10s linear infinite;
    }
    @keyframes spin {
        from { transform: translateX(-50%) rotate(0deg); }
        to { transform: translateX(-50%) rotate(360deg); }
    }
    /* Responsive adjustments */
    @media (max-width: 2400px) {
        .container {
            padding: 1rem;
        }
        header h1 {
            font-size: 2rem;
        }
        .bin-container {
            gap: 0.5rem;
        }
        .recycling-bin {
            width: 100px;
        }
    }
    /* Custom styles for upload area */
    .upload-area {
        border: 2px dashed var(--primary);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        background-color: rgba(76, 175, 80, 0.05);
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        background-color: rgba(76, 175, 80, 0.1);
    }
    .upload-icon {
        font-size: 2rem;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    </style>
    """
    font_awesome = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """
    st.markdown(css + font_awesome, unsafe_allow_html=True)

def main():
    # Load model
    model = load_model()
    
    # Embed CSS
    embed_css()

    # Header
    st.markdown("""
    <header>
      <div class="icon-large recycling-icon">
        <i class="fas fa-recycle"></i>
      </div>
      <h1>EcoSort Guardian</h1>
      <div class="subtitle">Smart recycling photo analyzer and sorting assistant</div>
    </header>
    """, unsafe_allow_html=True)

    # Main Content
    col1, col2 = st.columns([2, 1])

    # Left Column (Col1)
    with col1:
        # Photo Scanner Card
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <div class="icon-circle">
              <i class="fas fa-recycle"></i>
            </div>
            <h2>Photo Scanner</h2>
          </div>
          <div class="card-content">
        """, unsafe_allow_html=True)
        
        # Upload area with custom styling
        st.markdown("""
        <div class="upload-area">
          <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
          </div>
          <p>Upload an image of waste to classify</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add camera input option
        img_source = st.radio("Choose input source:", ["Upload Image", "Take Photo"])
        
        if img_source == "Upload Image":
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader", label_visibility="collapsed")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            camera_input = st.camera_input("Take a photo")
            if camera_input is not None:
                image = Image.open(camera_input)
        
        if 'image' in locals():
            # Display the image
            st.image(image, caption='Captured Image', use_container_width=True)
            
            # Make prediction
            predicted_idx = predict_image(image, model)
            predicted_class = class_names[predicted_idx]
            bin_category = class_to_bin[predicted_class]
            
            # Determine bin color for display
            if "Yellow" in bin_category:
                bin_color = "yellow"
            else:
                bin_color = "black"
            
            # Show results
            st.markdown(f"""
            <div class="detection-result">
              <div class="detection-item">
                <div class="detection-header {bin_color}">
                  <span>{predicted_class.capitalize()} - {bin_category.split('(')[1].replace(')', '')}</span>
                </div>
                <div class="detection-info">
                  <p>This item goes in the <strong>{bin_category}</strong></p>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)  # Close card tags

        # Alert Card
        st.markdown("""
        <div class="alert-card">
          <div class="alert-icon">
            <i class="fas fa-info-circle"></i>
          </div>
          <div class="alert-content">
            <h3>Smart Recycling System</h3>
            <p>This system detects and classifies waste into different categories, helping you sort items 
              into the correct recycling bins:</p>
            <div class="bin-categories">
              <div class="bin-category">
                <div class="color-dot yellow"></div>
                <span class="bin-name">Yellow Bin - Recyclables:</span>
                <span class="bin-desc">Plastic, cardboard, paper, metal, and glass items</span>
              </div>
              <div class="bin-category">
                <div class="color-dot black"></div>
                <span class="bin-name">Black Bin - General Waste:</span>
                <span class="bin-desc">Food scraps, Organic Wastes</span>
              </div>
            </div>
            <div class="recycling-message">
              <i class="fas fa-recycle"></i>
              <span>Proper recycling helps reduce landfill waste and conserves natural resources</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Right Column (Col2)
    with col2:
        # Available Recycling Bins Card
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <h2>Available Recycling Bins</h2>
          </div>
          <div class="card-content">
            <div class="bin-container">
              <div class="recycling-bin">
                <div class="bin yellow-bin">
                  <div class="bin-lid"></div>
                  <div class="bin-label">YELLOW</div>
                </div>
                <div class="bin-info">
                  <div class="bin-title">Yellow Bin</div>
                  <div class="bin-description">For all packaging and recyclables including plastic bottles, cans, cardboard, and paper.</div>
                </div>
              </div>
              <div class="recycling-bin">
                <div class="bin black-bin">
                  <div class="bin-lid"></div>
                  <div class="bin-label">BLACK</div>
                </div>
                <div class="bin-info">
                  <div class="bin-title">Black Bin</div>
                  <div class="bin-description">For general waste including food scraps.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Sorting Information Card
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <h2>Sorting Information</h2>
          </div>
          <div class="card-content">
            <div class="sorting-info">
              <h3>How to use this app:</h3>
              <p>Upload a photo of a waste item, and the app will tell you which bin it belongs in.</p>
              <ul>
                <li>Yellow bin - recyclables (plastic, paper, metal, glass, cardboard)</li>
                <li>Black bin - general waste (Organic Waste)</li>
              </ul>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()