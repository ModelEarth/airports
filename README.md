# airports

Run the airport pull using [data-pipeline/admin](https://model.earth/data-pipeline/admin/#node=airports)

[Airports sample map](https://model.earth/team/projects/map/#show=airports)

## Aviation emissions

The [emissions pipeline](emissions/) processes OpenSky flight CSVs using its own
[config.yaml](emissions/config.yaml) and exports daily estimated CO2 by flight,
route and directory airport. It runs locally without a cloud backend.
