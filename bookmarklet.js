javascript:(function(){
  // Set region name
  document.getElementById('region').value = 'Dummy Region ' + Math.floor(Math.random() * 100);
  
  // Years to fill
  const years = [2019, 2020, 2021, 2022, 2023];
  
  // Generate random values for each indicator and year
  years.forEach(year => {
    // Find all input fields for this year
    const inputs = document.querySelectorAll(`input[id^="value_${year}_"]`);
    
    // Fill each input with a random value
    inputs.forEach(input => {
      const indicator = input.id.split('_')[2]; // Extract indicator name
      let value;
      
      // Generate appropriate random values based on indicator type
      if (indicator.includes('persen') || 
          indicator === 'indeks_pembangunan_manusia' || 
          indicator === 'tingkat_pengangguran_terbuka' || 
          indicator === 'penduduk_miskin' || 
          indicator === 'angka_melek_huruf' || 
          indicator.includes('angka_partisipasi')) {
        // Percentage values (0-100)
        value = (Math.random() * 100).toFixed(2);
      } else if (indicator === 'pdrb_harga_konstan' || 
                indicator === 'daftar_upah_minimum') {
        // Large currency values
        value = (Math.random() * 10000000 + 1000000).toFixed(2);
      } else if (indicator === 'jml_pengeluaran_per_kapita') {
        // Medium currency values (thousands)
        value = (Math.random() * 1000 + 500).toFixed(2);
      } else if (indicator === 'jml_penduduk_bekerja') {
        // Population values
        value = (Math.random() * 1000000 + 10000).toFixed(0);
      } else if (indicator === 'kendaraan_roda_2' || 
                indicator === 'kendaraan_roda_4') {
        // Vehicle units
        value = (Math.random() * 100000 + 5000).toFixed(0);
      } else if (indicator === 'panjang_ruas_jalan') {
        // Road length in km
        value = (Math.random() * 1000 + 100).toFixed(2);
      } else if (indicator === 'titik_layanan_internet' || 
                indicator === 'kawasan_pariwisata') {
        // Locations/points
        value = (Math.random() * 100 + 5).toFixed(0);
      } else if (indicator === 'fasilitas_kesehatan') {
        // Health facilities
        value = (Math.random() * 50 + 5).toFixed(0);
      } else if (indicator.includes('kematian')) {
        // Death counts
        value = (Math.random() * 100).toFixed(0);
      } else if (indicator === 'angka_harapan_hidup') {
        // Life expectancy
        value = (Math.random() * 20 + 60).toFixed(2);
      } else if (indicator === 'rata_rata_lama_sekolah') {
        // School years
        value = (Math.random() * 6 + 6).toFixed(2);
      } else {
        // Default random value
        value = (Math.random() * 100).toFixed(2);
      }
      
      // Set the value
      input.value = value;
    });
  });
  
  // Alert when done
  alert('Form filled with dummy values!');
})();
