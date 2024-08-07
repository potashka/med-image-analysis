document.getElementById("upload-form").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData();
    const image = document.getElementById("image").files[0];
    formData.append("image", image);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById("results");
        resultsDiv.innerHTML = `
            <h2>Results:</h2>
            <p><strong>DeepHealth:</strong> ${data.DeepHealth}</p>
            <p><strong>DeepMED:</strong> ${data.DeepMED}</p>
            <p><strong>NiftyNet:</strong> ${data.NiftyNet}</p>
        `;
    })
    .catch(error => {
        console.error("Error:", error);
    });
});
