document.addEventListener("DOMContentLoaded", function () {
  // 1. Efek Visual saat Checkbox diklik
  const checkboxes = document.querySelectorAll(".form-check-input");

  checkboxes.forEach(function (checkbox) {
    checkbox.addEventListener("change", function () {
      // Cari elemen induk (div .gejala-box)
      const parentBox = this.closest(".gejala-box");

      if (this.checked) {
        parentBox.classList.add("active");
      } else {
        parentBox.classList.remove("active");
      }
    });
  });

  // 2. Validasi Form Sebelum Submit
  const form = document.querySelector("form");
  form.addEventListener("submit", function (event) {
    // Hitung berapa checkbox yang dicentang
    const checkedCount = document.querySelectorAll(
      'input[name="gejala"]:checked'
    ).length;

    if (checkedCount === 0) {
      // Stop proses submit
      event.preventDefault();

      // Tampilkan peringatan
      alert("⚠️ Mohon pilih setidaknya satu gejala yang Anda rasakan!");
    }
  });
});
