// Custom JavaScript for Prosperity Analysis Application

document.addEventListener("DOMContentLoaded", function () {
  // Initialize tooltips
  var tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Flash message auto-dismiss
  setTimeout(function () {
    var alerts = document.querySelectorAll(".alert-dismissible");
    alerts.forEach(function (alert) {
      var bsAlert = new bootstrap.Alert(alert);
      bsAlert.close();
    });
  }, 5000);

  // Confirm delete modal functionality
  var deleteButtons = document.querySelectorAll(".btn-delete");
  deleteButtons.forEach(function (button) {
    button.addEventListener("click", function (e) {
      e.preventDefault();
      var dataId = this.getAttribute("data-id");
      var deleteForm = document.getElementById("deleteForm");
      if (deleteForm) {
        deleteForm.action = deleteForm.action.replace("__id__", dataId);
        var deleteModal = new bootstrap.Modal(
          document.getElementById("deleteModal")
        );
        deleteModal.show();
      }
    });
  });

  // Model selection in visualization
  var modelButtons = document.querySelectorAll(".model-select-btn");
  modelButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      var modelType = this.getAttribute("data-model");
      var form = document.getElementById("modelSelectionForm");
      if (form) {
        var modelInput = form.querySelector('input[name="model_type"]');
        modelInput.value = modelType;
        form.submit();
      }
    });
  });

  // Visualization type change
  var vizTypeSelect = document.getElementById("viz_type");
  if (vizTypeSelect) {
    vizTypeSelect.addEventListener("change", function () {
      this.form.submit();
    });
  }

  // Year and indicator selectors
  var yearSelect = document.getElementById("year");
  var indicatorSelect = document.getElementById("indicator");

  if (yearSelect) {
    yearSelect.addEventListener("change", function () {
      this.form.submit();
    });
  }

  if (indicatorSelect) {
    indicatorSelect.addEventListener("change", function () {
      this.form.submit();
    });
  }

  // Province autocomplete for add data form
  var provinceInput = document.getElementById("province");
  if (provinceInput) {
    // This would be replaced with actual province data from the backend
    // For now, we'll use a placeholder array
    var provinces = [
      "Aceh",
      "Sumatera Utara",
      "Sumatera Barat",
      "Riau",
      "Jambi",
      "Sumatera Selatan",
      "Bengkulu",
      "Lampung",
      "Kepulauan Bangka Belitung",
      "Kepulauan Riau",
      "DKI Jakarta",
      "Jawa Barat",
      "Jawa Tengah",
      "DI Yogyakarta",
      "Jawa Timur",
      "Banten",
      "Bali",
      "Nusa Tenggara Barat",
      "Nusa Tenggara Timur",
      "Kalimantan Barat",
      "Kalimantan Tengah",
      "Kalimantan Selatan",
      "Kalimantan Timur",
      "Kalimantan Utara",
      "Sulawesi Utara",
      "Sulawesi Tengah",
      "Sulawesi Selatan",
      "Sulawesi Tenggara",
      "Gorontalo",
      "Sulawesi Barat",
      "Maluku",
      "Maluku Utara",
      "Papua Barat",
      "Papua",
    ];

    // Simple autocomplete functionality
    provinceInput.addEventListener("input", function () {
      var val = this.value.toLowerCase();
      var datalist = document.getElementById("provinceList");

      // Clear existing options
      datalist.innerHTML = "";

      // Add matching provinces
      provinces.forEach(function (province) {
        if (province.toLowerCase().includes(val)) {
          var option = document.createElement("option");
          option.value = province;
          datalist.appendChild(option);
        }
      });
    });
  }

  // Form validation
  var forms = document.querySelectorAll(".needs-validation");
  Array.prototype.slice.call(forms).forEach(function (form) {
    form.addEventListener(
      "submit",
      function (event) {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        }
        form.classList.add("was-validated");
      },
      false
    );
  });
});
