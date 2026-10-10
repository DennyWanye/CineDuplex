// Continuous, band-limited conversion to the model's 16kHz PCM input.
// Keep the playback AudioContext at the device rate; never gate microphone audio.
class Capture extends AudioWorkletProcessor {
  constructor(){
    super();this.half=31;this.step=sampleRate/16000;
    this.buffer=Array(this.half).fill(0);this.pos=this.half;this.samples=[];this.meter=0;
    this.filters=[];
    const cutoff=Math.min(.45,7200/sampleRate);
    for(let phase=0;phase<256;phase++){
      const fraction=phase/256,coefficients=[];let total=0;
      for(let j=-this.half;j<=this.half;j++){
        const t=j-fraction,z=2*cutoff*t;
        const sinc=Math.abs(z)<1e-12?1:Math.sin(Math.PI*z)/(Math.PI*z);
        const window=.42+.5*Math.cos(Math.PI*j/this.half)+.08*Math.cos(2*Math.PI*j/this.half);
        const value=2*cutoff*sinc*window;coefficients.push(value);total+=value;
      }
      this.filters.push(coefficients.map(value=>value/total));
    }
  }
  process(inputs,outputs){
    for(const ch of outputs[0]||[])ch.fill(0);
    const channels=inputs[0];if(!channels?.length)return true;
    const frames=channels[0].length;
    for(let i=0;i<frames;i++){let v=0;for(const ch of channels)v+=ch[i]/channels.length;this.buffer.push(v);}
    while(this.pos+this.half<this.buffer.length){
      const center=Math.floor(this.pos),phase=Math.min(255,Math.floor((this.pos-center)*256));
      const filter=this.filters[phase];let v=0;
      for(let j=-this.half;j<=this.half;j++)v+=this.buffer[center+j]*filter[j+this.half];
      this.samples.push(Math.max(-32768,Math.min(32767,Math.round(v*32768))));this.meter+=v*v;this.pos+=this.step;
      if(this.samples.length===6400){
        const bytes=new ArrayBuffer(12800),view=new DataView(bytes);
        this.samples.forEach((x,i)=>view.setInt16(i*2,x,true));
        this.port.postMessage({pcm:bytes,rms:Math.sqrt(this.meter/6400)},[bytes]);this.samples=[];this.meter=0;
      }
    }
    const used=Math.max(0,Math.floor(this.pos)-this.half);
    this.buffer.splice(0,used);this.pos-=used;return true;
  }
}
registerProcessor('continuous-capture',Capture);
