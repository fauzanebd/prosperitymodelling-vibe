// Global prosperity data variable
let prosperityData = {};

// Initialize the map
function initMap() {
  // Create a map centered on West Java
  const map = L.map("west-java-map").setView([-6.9, 107.6], 8);

  // Load prosperity data from data attribute
  const mapElement = document.getElementById("west-java-map");
  if (mapElement && mapElement.dataset.prosperity) {
    try {
      prosperityData = JSON.parse(mapElement.dataset.prosperity);
      console.log(
        "Loaded prosperity data:",
        Object.keys(prosperityData).length,
        "regions"
      );
    } catch (e) {
      console.error("Error parsing prosperity data:", e);
    }
  }

  // Add OpenStreetMap tile layer
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  // Show loading indicator
  const loadingIndicator = document.querySelector(".map-loading");
  if (loadingIndicator) {
    loadingIndicator.classList.remove("d-none");
  }

  // Load GeoJSON data
  fetch("/static/data/west_java.geojson")
    .then((response) => response.json())
    .then((data) => {
      // Hide loading indicator
      if (loadingIndicator) {
        loadingIndicator.classList.add("d-none");
      }

      // Add GeoJSON layer
      const geojsonLayer = L.geoJSON(data, {
        style: function (feature) {
          return {
            fillColor: getRegionColor(
              feature.properties.name,
              feature.properties
            ),
            weight: 1.5,
            opacity: 1,
            color: "#555",
            dashArray: "",
            fillOpacity: 0.7,
          };
        },
        onEachFeature: function (feature, layer) {
          // Add mouseover events for interactivity
          layer.on({
            mouseover: highlightFeature,
            mouseout: resetHighlight,
            click: zoomToFeature,
          });

          // Format region name to display properly (handle both KotaBandung and Kota Bandung formats)
          const regionName =
            feature.properties.name || feature.properties.NAME_2;
          const formattedRegionName = formatRegionName(
            regionName,
            feature.properties
          );

          // Get prosperity class for the region
          const prosperityClass = getRegionProsperityClass(
            formattedRegionName,
            feature.properties
          );
          const prosperityClassColor = getProsperityClassColor(prosperityClass);

          // Add popup with region info
          const popupContent = `
                        <div class="region-popup">
                            <h5>${formattedRegionName}</h5>
                            <p>Status: <span style="color: ${prosperityClassColor}"><strong>${prosperityClass}</strong></span></p>
                        </div>
                    `;
          layer.bindPopup(popupContent);

          // Add a tooltip for quick hover info
          layer.bindTooltip(formattedRegionName, {
            permanent: false,
            direction: "center",
            className: "region-tooltip",
          });
        },
      }).addTo(map);

      // Fit the map to the bounds of West Java
      map.fitBounds(geojsonLayer.getBounds());

      // Add a legend
      addLegend(map);
    })
    .catch((error) => {
      console.error("Error loading GeoJSON:", error);
      // Hide loading indicator and show error message
      if (loadingIndicator) {
        loadingIndicator.classList.add("d-none");
      }

      // Display error message on the map container
      const mapContainer = document.getElementById("west-java-map");
      if (mapContainer) {
        mapContainer.innerHTML = `
                    <div class="alert alert-danger mt-3" role="alert">
                        <strong>Error loading map data:</strong> ${
                          error.message || "Unknown error"
                        }
                    </div>
                `;
      }
    });
}

// Format region name for display (e.g., convert "KotaBandung" to "Kota Bandung")
function formatRegionName(rawName, properties) {
  if (!rawName) return "Unknown Region";

  // Check if we have the TYPE_2 field to determine if it's a city (Kota) or regency (Kabupaten)
  if (properties && properties.TYPE_2) {
    const type = properties.TYPE_2;

    // If name already has the type prefix, just format it properly
    if (rawName.toLowerCase().startsWith(type.toLowerCase())) {
      // Name already has prefix, just ensure proper spacing and capitalization
      if (type.toLowerCase() === "kota") {
        return "Kota " + rawName.substring(4);
      } else if (type.toLowerCase() === "kabupaten") {
        return "Kabupaten " + rawName.substring(9);
      }
    } else {
      // Name doesn't have prefix, add it
      if (type.toLowerCase() === "kota") {
        return "Kota " + rawName;
      } else if (type.toLowerCase() === "kabupaten") {
        return "Kabupaten " + rawName;
      }
    }
  }

  // Fallback to existing formatting logic
  // Check if the name already has spaces
  if (rawName.includes(" ")) return rawName;

  // Special prefixes to handle
  const prefixes = ["Kota", "Kabupaten", "Waduk"];

  // Check for prefixes
  for (const prefix of prefixes) {
    if (rawName.startsWith(prefix) && rawName.length > prefix.length) {
      // Add a space after the prefix
      return prefix + " " + rawName.substring(prefix.length);
    }
  }

  // For other cases, add spaces before capital letters (CamelCase to "Camel Case")
  return rawName.replace(/([A-Z])/g, " $1").trim();
}

// Function to get prosperity class for a region
function getRegionProsperityClass(regionName, properties) {
  // Handle common formatting differences in region names
  const normalizedRegionName = normalizeRegionName(regionName);

  // Check if we have data for this region
  if (prosperityData) {
    // Try exact match first
    if (prosperityData[regionName]) {
      return prosperityData[regionName];
    }

    // Format the region name with properties and try matching
    if (properties) {
      const formattedRegionName = formatRegionName(regionName, properties);
      if (prosperityData[formattedRegionName]) {
        return prosperityData[formattedRegionName];
      }
    }

    // Try searching in our normalized region names
    for (const key in prosperityData) {
      const normalizedKey = normalizeRegionName(key);
      if (normalizedKey === normalizedRegionName) {
        return prosperityData[key];
      }
    }
  }

  return "Data tidak tersedia";
}

// Normalize region name to handle different formatting (e.g., "Kota Bandung" vs "KotaBandung")
function normalizeRegionName(name) {
  if (!name) return "";

  // Remove spaces, convert to lowercase
  return name.replace(/\s+/g, "").toLowerCase();
}

// Function to get color based on prosperity class
function getRegionColor(regionName, properties) {
  const prosperityClass = getRegionProsperityClass(regionName, properties);
  return getProsperityClassColor(prosperityClass);
}

// Get color for prosperity class
function getProsperityClassColor(prosperityClass) {
  switch (prosperityClass) {
    case "Sejahtera":
      return "#28a745"; // success color - green
    case "Menengah":
      return "#ffc107"; // warning color - yellow
    case "Tidak Sejahtera":
      return "#dc3545"; // danger color - red
    default:
      return "#6c757d"; // secondary color - gray for no data
  }
}

// Highlight feature on hover
function highlightFeature(e) {
  const layer = e.target;

  layer.setStyle({
    weight: 3,
    color: "#333",
    dashArray: "",
    fillOpacity: 0.9,
  });

  layer.bringToFront();

  // Update info panel if it exists
  if (window.updateInfoPanel) {
    window.updateInfoPanel(layer.feature.properties);
  }
}

// Reset highlight on mouseout
function resetHighlight(e) {
  const layer = e.target;

  layer.setStyle({
    weight: 1.5,
    opacity: 1,
    color: "#555",
    dashArray: "",
    fillOpacity: 0.7,
  });
}

// Zoom to feature on click
function zoomToFeature(e) {
  map.fitBounds(e.target.getBounds());
}

// Add a legend to the map
function addLegend(map) {
  const legend = L.control({ position: "bottomright" });

  legend.onAdd = function (map) {
    const div = L.DomUtil.create("div", "info legend");
    const grades = [
      "Sejahtera",
      "Menengah",
      "Tidak Sejahtera",
      "Data tidak tersedia",
    ];
    const colors = ["#28a745", "#ffc107", "#dc3545", "#6c757d"];

    div.innerHTML = "<h6>Tingkat Kesejahteraan</h6>";

    for (let i = 0; i < grades.length; i++) {
      div.innerHTML +=
        '<div class="legend-item">' +
        '<i style="background:' +
        colors[i] +
        '"></i> ' +
        grades[i] +
        "</div>";
    }

    return div;
  };

  legend.addTo(map);
}

// Initialize map when document is ready
document.addEventListener("DOMContentLoaded", function () {
  if (document.getElementById("west-java-map")) {
    initMap();
  }
});
