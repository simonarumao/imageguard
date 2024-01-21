const uploadForm = document.getElementById("upload-form");
const imageUpload = document.getElementById("image-upload");
const imagePreview = document.getElementById("image-preview");

imageUpload.addEventListener("change", handleImageUpload);

function handleImageUpload() {
  imagePreview.innerHTML = "";
  const files = imageUpload.files;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.classList.add("preview-image");
      imagePreview.appendChild(img);
    } else {
      alert("Please select an image file.");
    }
  }
}

uploadForm.addEventListener("submit", function (e) {
  e.preventDefault();
});
