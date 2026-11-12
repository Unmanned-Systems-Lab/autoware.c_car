"""Reject hardware-interface launches for this simulation-only vehicle."""


def generate_launch_description():
    raise RuntimeError(
        "c_car_vehicle_launch is simulation-only. "
        "Use vehicle_simulation:=true and launch_vehicle_interface:=false. "
        "A C-Car hardware interface has not been configured."
    )
