File "/home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 88, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 590, in code_to_exec
    exec(code, module.__dict__)
File "/mount/src/lenskart_dash/app.py", line 123, in <module>
    st.plotly_chart(fig_bubble, use_container_width=True)
File "/home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/metrics_util.py", line 410, in wrapped_func
    result = non_optional_func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/streamlit/elements/plotly_chart.py", line 493, in plotly_chart
    plotly_chart_proto.spec = plotly.io.to_json(figure, validate=False)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/plotly/io/_json.py", line 221, in to_json
    return to_json_plotly(fig_dict, pretty=pretty, engine=engine)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/plotly/io/_json.py", line 142, in to_json_plotly
    json.dumps(plotly_object, cls=PlotlyJSONEncoder, **opts), _swap_json
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
          ^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/_plotly_utils/utils.py", line 56, in encode
    encoded_o = super(PlotlyJSONEncoder, self).encode(o)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/json/encoder.py", line 200, in encode
    chunks = self.iterencode(o, _one_shot=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/json/encoder.py", line 258, in iterencode
    return _iterencode(o, 0)
           ^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/_plotly_utils/utils.py", line 133, in default
    return _json.JSONEncoder.default(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
