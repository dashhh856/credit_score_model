
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('predictionText');
    const probabilityText = document.getElementById('probabilityText');
    
    // UI Loading state
    submitBtn.classList.add('loading');
    resultDiv.classList.add('hidden');
    resultDiv.className = 'result hidden'; // Reset classes
    
    // Collect JSON
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields from string to number
    // A2, A3, A8, A11, A14, A15 are numeric in our Pydantic model
    const numericFields = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15'];
    numericFields.forEach(field => {
        if (data[field]) {
            data[field] = parseFloat(data[field]);
        }
    });

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Display Result
        resultDiv.classList.remove('hidden');
        
        const isApproved = result.prediction === '+' || result.prediction === 'Approved';
        
        if (isApproved) {
            resultDiv.classList.add('approved');
            predictionText.textContent = 'Application Approved';
            predictionText.style.color = '#4ade80';
        } else {
            resultDiv.classList.add('rejected');
            predictionText.textContent = 'Application Rejected';
            predictionText.style.color = '#f87171';
        }
        
        // Add fake probability if not provided, or display meaningful message
        probabilityText.textContent = "Based on the provided financial and demographic data.";
        
    } catch (error) {
        alert("Failed to get prediction: " + error.message);
        console.error(error);
    } finally {
        submitBtn.classList.remove('loading');
    }
});
