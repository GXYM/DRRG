import argparse

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description="test global argument parser")

# script parameters
parser.add_argument('-g', '--GT_PATH', default='gt/gt.zip', help="Path of the Ground Truth file.")
parser.add_argument('-s', '--SUBMIT_PATH', default='submit.zip', help="Path of your method's results file.")

# webserver parameters
parser.add_argument('-o', '--OUTPUT_PATH', default='output/', help="Path to a directory where to copy the file that contains per-sample results.")
parser.add_argument('-p', '--PORT', default=8080, help='port number to show')
parser.add_argument('--HOST', default='0.0.0.0', help='port number to show')

# result format related parameters
parser.add_argument('--BOX_TYPE', default='QUAD', choices=['LTRB', 'QUAD', 'POLY'])
parser.add_argument('--TRANSCRIPTION', action='store_true')
parser.add_argument('--CONFIDENCES', action='store_true')
parser.add_argument('--CRLF', action='store_true')

# end-to-end related parameters
parser.add_argument('--E2E', action='store_true')
parser.add_argument('--CASE_SENSITIVE', default=True, type=str2bool)
parser.add_argument('--RS', default=True, type=str2bool)

# other parameters
parser.add_argument('-t', '--NUM_WORKERS', default=2, type=int, help='number of threads to use')
parser.add_argument('--GT_SAMPLE_NAME_2_ID', default='([0-9]+)')
parser.add_argument('--DET_SAMPLE_NAME_2_ID', default='([0-9]+)')
parser.add_argument('--PER_SAMPLE_RESULTS', default=True, type=str2bool)

# PARAMS = parser.parse_args()
PARAMS, unknown = parser.parse_known_args()
