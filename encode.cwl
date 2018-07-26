$namespaces:
  arv: "http://arvados.org/cwl#"
  cwltool: "http://commonwl.org/cwltool#"
cwlVersion: v1.0
class: CommandLineTool
requirements:
  - class: DockerRequirement
    dockerPull: kfang/scitool_old
  - class: InlineJavascriptRequirement
  - class: ResourceRequirement
    ramMin: 150000
    coresMin: 4 
stdout: $("encode_output.txt")
hints:
  arv:RuntimeConstraints:
    keep_cache: 1500
baseCommand: python
inputs:
  script:
    type: File
    inputBinding:
      position: 0
  arr:
    type: File
    inputBinding:
      position: 1
outputs:
  enc_out:
    type: stdout
  out2:
    type: File[]
    outputBinding:
      glob: "*npy"
