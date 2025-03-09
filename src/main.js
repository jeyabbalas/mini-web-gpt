import './style.css'
import {Tensor} from "../lib/main.js";


const t = new Tensor([2, 3], true);

document.querySelector('#app').innerHTML = `
  <div>
    <p>${t}</p>
  </div>
`

