const playIcon = document.getElementById("playIcon");
const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const ctx = cameraCanvas.getContext('2d');
const recognizedTextDiv = document.getElementById("recognizedText");


// Función de texto a voz
function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'es-ES'; 
        window.speechSynthesis.speak(utterance);
    } else {
        console.log("API de text-to-speech no soportada en este navegador.");
    }
}

// Simulación de reconocimiento de lenguaje de señas
function recognizeSignLanguage() {
    const detectedText = "Bienvenido a HandSpeak AI"; 
    recognizedTextDiv.textContent = detectedText;     
    speakText(detectedText);                          
}


// Función para iniciar la cámara
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraVideo.srcObject = stream;
        cameraVideo.style.display = "block";  
        playIcon.style.display = "none";      
        requestAnimationFrame(drawVideoOnCanvas);
    } catch (error) {
        console.error("Error al acceder a la cámara:", error);
    }
}

// Función para dibujar el video dentro del canvas
function drawVideoOnCanvas() {
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    requestAnimationFrame(drawVideoOnCanvas);
}

playIcon.addEventListener("click", startCamera);