// Simple debug version to test in console
// First, set the region name
console.log("Starting form fill...");
const regionField = document.getElementById('region');
if (regionField) {
  regionField.value = 'Dummy Region ' + Math.floor(Math.random() * 100);
  console.log("Region field filled");
} else {
  console.error("Region field not found!");
}

// Try to find all input fields for a specific year (e.g., 2019)
const year2019Inputs = document.querySelectorAll('input[id^="value_2019_"]');
console.log("Found " + year2019Inputs.length + " inputs for 2019");

// If inputs were found, try to fill the first one
if (year2019Inputs.length > 0) {
  const firstInput = year2019Inputs[0];
  const indicator = firstInput.id.split('_')[2];
  console.log("First indicator: " + indicator);
  firstInput.value = "75.5";
  console.log("First input filled with 75.5");
} else {
  console.error("No inputs found for 2019!");
}

// Log all form inputs to see what's available
console.log("All form inputs:");
const allInputs = document.querySelectorAll('input');
allInputs.forEach(input => {
  console.log(input.id + " - type: " + input.type);
});
