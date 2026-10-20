# Google Colab Setup & Training Guide (मराठी & English)
> **Multi-Crop Monitoring System** — Phase 3A EfficientNet-B0 Disease Classifiers  
> *Google Colab वर T4 GPU चा वापर करून मॉडेल ट्रेनिंग आणि टेस्ट इव्हॅल्युएशन करण्याची मार्गदर्शिका.*

---

## 1. कोणत्या फाइल्स झिप (ZIP) करून Google Drive वर अपलोड करायच्या? (Dataset Upload Requirements)

आपल्या लॅपटॉपवर GPU नसल्यामुळे आणि डिस्क स्पेस मर्यादित असल्यामुळे प्रत्यक्ष मॉडेल ट्रेनिंग **Google Colab (T4 GPU)** वर केले जाते. मूळ डेटासेट्स झिप करून Google Drive वर अपलोड करावे लागतात.

> [!IMPORTANT]
> मूळ इमेज फाइल्स किंवा लेबल फाइल्समध्ये कोणताही बदल करू नका. प्रत्येक डेटासेट फोल्डरच्या **आत जाऊन त्यातील घटक (CONTENTS)** स्वतंत्रपणे ZIP करा (पालक फोल्डर नव्हे, जेणेकरून ZIP च्या शीर्षस्थानी थेट क्लास फोल्डर्स असतील):

| पीक (Crop) | डेटासेट नाव (Dataset Folder) | एकूण प्रतिमा (Images) | प्रत्यक्ष ZIP आकार (Real ZIP Size) | क्लासेस (Classes) |
| :--- | :--- | :---: | :---: | :--- |
| **ऊस (Sugarcane)** | `sugarcane_sppu` | 2,521 | **167,253,039 bytes** | `Healthy`, `Mosaic`, `RedRot`, `Rust`, `Yellow` |
| **गहू (Wheat)** | `wheat_zenodo_jic` | 999 | **1,029,392,786 bytes** | `BrownRust`, `Healthy`, `Mildew`, `Septoria`, `YellowRust` |
| **कापूस (Cotton)** | `cotton_mendeley` | 980 | **111,575,900 bytes** | `bacterial blight`, `curl virus`, `fussarium wilt`, `healthy` |
| **भात (Rice)** | `rice_pest_lakthuru` | 5,229 | **195,886,432 bytes** | 6 pest classes |

### ZIP तयार करण्यासाठी अचूक Windows PowerShell कमांड्स (Zip CONTENTS, not folder):
```powershell
# E:\datasets_raw मधील प्रत्येक फोल्डरच्या आत जाऊन त्यातील घटक (CONTENTS) ZIP करा:
Set-Location "E:\datasets_raw\sugarcane_sppu"
Compress-Archive -Path * -DestinationPath "E:\temp\sugarcane_sppu.zip"

Set-Location "E:\datasets_raw\wheat_zenodo_jic"
Compress-Archive -Path * -DestinationPath "E:\temp\wheat_zenodo_jic.zip"

Set-Location "E:\datasets_raw\cotton_mendeley"
Compress-Archive -Path * -DestinationPath "E:\temp\cotton_mendeley.zip"

Set-Location "E:\datasets_raw\rice_pest_lakthuru"
Compress-Archive -Path * -DestinationPath "E:\temp\rice_pest_lakthuru.zip"
```

---

## 2. Google Drive फोल्डर रचना (Google Drive Folder Hierarchy)

आपल्या मुख्य Google Drive वर खालीलप्रमाणे फोल्डर तयार करा आणि ZIP फाइल्स तिथे अपलोड करा:

```text
Google Drive (MyDrive)/
└── multicrop/
    ├── datasets/                     <-- येथे 3 ZIP फाइल्स अपलोड करा
    │   ├── sugarcane_sppu.zip
    │   ├── wheat_zenodo_jic.zip
    │   └── cotton_mendeley.zip
    ├── checkpoints/                  <-- ट्रेनिंग दरम्यान वेट्स (.pt) आपोआप येथे सेव्ह होतील
    │   ├── sugarcane/
    │   ├── wheat/
    │   └── cotton/
    └── reports/                      <-- टेस्ट इव्हॅल्युएशन JSON आणि PNG चार्ट्स येथे सेव्ह होतील
```

---

## 3. Google Colab वर T4 GPU सेटअप करण्याचे टप्पे (Step-by-Step Execution on T4 GPU)

### टप्पा १: GPU ॲक्सिलरेटर निवडा (Select T4 GPU)
1. [Google Colab](https://colab.research.google.com/) उघडा.
2. `File` $\to$ `Upload notebook` वर क्लिक करून `notebooks/colab_train_classifiers.ipynb` निवडा.
3. मेन्यू बारमधून **Runtime** $\to$ **Change runtime type** निवडा.
4. **Hardware accelerator** मध्ये **T4 GPU** निवडा आणि **Save** करा.

### टप्पा २: Google Drive माउंट करा आणि कोड रन करा (Mount Drive & Run)
1. **Step 1 सेल**: Google Drive कनेक्ट करण्यासाठी ऑथोरायझेशन द्या.
2. **Step 2 सेल**: गिट रिपॉझिटरी क्लोन करा (`git clone ...`).
3. **Step 3 सेल**: आवश्यक पॅकेजेस इन्स्टॉल करा (`pip install timm torch ...`).
4. **Step 4 सेल**: डेटासेट ZIP फाइल्स Drive वरून थेट `/content/data` मध्ये अनझिप (extract) होतील.

---

## 4. Colab Disconnect झाल्यास काय करावे? (Resumption after Disconnection)

Colab चे मोफत सत्र अचानक डिस्कनेक्ट झाले तरी काळजी करण्याचे कारण नाही:

- आमची स्क्रिप्ट प्रत्येक epoch नंतर `best_checkpoint.pt` आणि `last_checkpoint.pt` थेट तुमच्या **Google Drive** वर सेव्ह करते.
- जेव्हा तुम्ही Colab पुन्हा सुरू कराल आणि `--resume` फ्लॅगसह कमांड रन कराल, तेव्हा मॉडेल जिथून थांबले होते अगदी त्याच epoch आणि learning rate पासून आपोआप पुढे सुरू होईल!
- उदाहरण: जर १५व्या epoch ला कनेक्शन तुटले, तर पुन्हा रन केल्यावर मॉडेल १६व्या epoch पासून सुरू होईल.

```bash
# Resumption command example (used automatically in the notebook):
python ml_training/scripts/train_classifier.py \
    --crop sugarcane \
    --data-root /content/data \
    --out-dir /content/drive/MyDrive/multicrop/checkpoints/sugarcane \
    --epochs 30 \
    --resume
```

---

## 5. मॉडेल ट्रेनिंगची वैशिष्ट्ये (Model Training Architecture & Governance)

1. **EfficientNet-B0 (Pretrained)**:
   - इमेजनेट प्री-ट्रेन्ड बॅकबोन वापरून वेगाने आणि अचूकतेने शिकते.
   - इनपुट आकार: $224 \times 224$ px.
2. **ऑन-द-फ्लाय ऑगमेंटेशन (Data Augmentation on Train Only)**:
   - केवळ `train` स्प्लिटवरच फिरवणे (horizontal/vertical flip), फिरकी (rotation 15°), रंग बदल (color jitter) आणि random resized crop केले जाते.
   - ऑगमेंट केलेल्या इमेजेस डिस्कवर सेव्ह केल्या जात नाहीत; त्यामुळे जागा वाचते.
   - `val` आणि `test` वर कोणतीही ऑगमेंटेशन केली जात नाही (केवळ standard center-crop & resize).
3. **ऊस डेटासेटसाठी ड्युप्लिकेशन फिल्टर (`use_for_eval`)**:
   - व्हॅलिडेशन आणि टेस्ट इव्हॅल्युएशन केवळ `use_for_eval == true` असलेल्या प्रतिमांवरच होते, जेणेकरून डुप्लिकेट इमेजेसमुळे मेट्रिक्स फुगणार नाहीत.
4. **कापूस डेटासेटसाठी विशेष उपाययोजना (Cotton Class Imbalance & 5-Fold CV)**:
   - अत्यंत कमी सॅम्पल्स असलेल्या रोगांसाठी (`curl virus`: train 18, `fussarium wilt`: train 33) **Train-split class weights** लॉस फंक्शनमध्ये वापरले आहेत.
   - `train+val` (882 प्रतिमा) वर **Stratified 5-Fold Cross Validation** चालवून per-class recall चा `mean ± std` रिपोर्ट केला जातो.
   - फायनल टेस्ट निकाल `indicative only (tiny classes: curl virus test n=2, fusarium wilt test n=5)` म्हणून लेबल केला जातो.
5. **विल्सन ९५% विश्वासार्हता मर्यादा (Wilson 95% Confidence Intervals)**:
   - प्रत्येक क्लासच्या recall साठी Wilson score interval संगणित केला जातो, ज्यामुळे कमी सॅम्पल्सच्या मर्यादा स्पष्ट दिसतात.

---

## 6. रिझल्ट्स लॅपटॉपवर परत आणणे (Syncing Results to Local Laptop)

ट्रेनिंग आणि इव्हॅल्युएशन पूर्ण झाल्यानंतर:
1. Google Drive मधील `MyDrive/multicrop/checkpoints/` आणि `MyDrive/multicrop/reports/` फोल्डर्स डाउनलोड करा.
2. लॅपटॉपवर प्रोजेक्टमधील `checkpoints/` आणि `reports/` डिरेक्टरीमध्ये ते ठेवा.
3. मॉडेल रजिस्टर करण्यासाठी कमांड रन करा:
```powershell
py ml_training/scripts/register_model.py `
    --checkpoint checkpoints/sugarcane/best_checkpoint.pt `
    --metrics reports/metrics_sugarcane_<timestamp>.json `
    --crop sugarcane
```
*(टीप: प्रत्यक्ष ट्रेनिंग पूर्ण होईपर्यंत रजिस्ट्रीमध्ये कोणतीही बनावट नोंद केली जात नाही).*
