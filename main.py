from flask import Flask, request, jsonify

from absl import app as absl_app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string ("host",  "localhost", "Host IP")
flags.DEFINE_integer("port", 6000,      "Host port")

app = Flask(__name__)

@app.route("/api/", methods=["GET"])
def index():
    """API Testing Endpoint."""
    return jsonify("Hello, World!")

def main(unused_argv):
    app.run(host=FLAGS.host, port=FLAGS.port)

def entry_point():
    absl_app.run(main)

if __name__ == "__main__":
    absl_app.run(main)