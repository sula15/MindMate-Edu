from pymilvus import connections

connections.connect(alias="default", host="localhost", port="19530")
print(connections.has_connection(alias="default"))
