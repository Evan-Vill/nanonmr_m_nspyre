# nspyre Template
An [nspyre](https://nspyre.readthedocs.io/en/latest/) experiment template.

To install, simply:
`pip install -e .`

To run the experiment, first start the instrument servers:
`python src/template/drivers/remote_inserv.py`
`python src/template/drivers/local_inserv.py`

Run the nspyre data server:
`nspyre-dataserv`

Finally, start the experiment GUI:
`python src/template/gui/app.py`
or
`template`
