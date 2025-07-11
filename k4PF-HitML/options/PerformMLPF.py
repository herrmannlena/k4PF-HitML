#
# Copyright (c) 2020-2024 Key4hep-Project.
#
# This file is part of Key4hep.
# See https://key4hep.github.io/key4hep-doc/ for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from Configurables import PFHitML
from Configurables import k4DataSvc
from Gaudi.Configuration import INFO
from k4FWCore import ApplicationMgr, IOSvc
from k4FWCore.parseArgs import parser

# parse the custom arguments
parser_group = parser.add_argument_group("PerformMLPF.py custom options")
parser_group.add_argument("--inputFiles", nargs="+", metavar=("file1", "file2"), help="One or multiple input files",
                        default=["/afs/cern.ch/work/l/lherrman/private/inference/data/Hss_rec_16809_40.root"])
#parser_group.add_argument("--outputFile", help="Output file name", default="output_MLPF.root")
parser_group.add_argument("--num_ev", type=int, help="Number of events to process (-1 means all)", default=-1)
parser_group.add_argument("--onnx_model_clustering", help="Path to ONNX model used for clustering", default="/eos/user/l/lherrman/FCC/models/clustering_1.onnx")
parser_group.add_argument("--json_onnx_config", help="Path to JSON config file for ONNX model used for clustering", default="/afs/cern.ch/work/l/lherrman/private/inference/k4PFHitML/scripts/config_hits_track_v2_noise.json")


args = parser.parse_known_args()[0]

svc = IOSvc("IOSvc")
svc.Input = args.inputFiles
#svc.Output = args.outputFile

Multitransformer = PFHitML("PFHitM",
                            model_path_clustering=args.onnx_model,
                            json_path=args.json_onnx_config
                    )

ApplicationMgr(TopAlg=[Multitransformer],
               EvtSel="NONE",
               EvtMax=args.num_ev,
               ExtSvc=[k4DataSvc("EventDataSvc")],
               OutputLevel=INFO,
               )

