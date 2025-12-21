async function loadSbox() {
  let name = document.getElementById("sbox").value;
  let res = await fetch(`/api/sbox/${name}`);
  let data = await res.json();

  let table = document.getElementById("sboxTable");
  table.innerHTML = "";

  data.matrix.forEach(row => {
    let tr = document.createElement("tr");
    row.forEach(v => {
      let td = document.createElement("td");
      td.innerText = v.toString(16).padStart(2,"0");
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
}

async function encrypt() {
  let res = await fetch("/api/encrypt", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      key: key.value,
      text: plain.value,
      sbox: sbox.value
    })
  });
  let data = await res.json();
  cipher.value = data.cipher;
}

async function decrypt() {
  let res = await fetch("/api/decrypt", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      key: key.value,
      cipher: cipher.value,
      sbox: sbox.value
    })
  });
  let data = await res.json();
  plain.value = data.plaintext;
}

fetch("/api/results")
  .then(r=>r.json())
  .then(d=>results.innerText = JSON.stringify(d,null,2));
