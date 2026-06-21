
from pydantic import BaseModel
from typing import Literal


AbyssCallouts = Literal["A Site", "B Site"]
class AbyssCalloutResult(BaseModel):
    assigned_label: AbyssCallouts | None


AscentCallouts = Literal["A Site", "B Site"]
class AscentCalloutResult(BaseModel):
    assigned_label: AscentCallouts | None


BindCallouts = Literal["A Site", "B Site"]
class BindCalloutResult(BaseModel):
    assigned_label: BindCallouts | None


BreezeCallouts = Literal["A Site", "B Site"]
class BreezeCalloutResult(BaseModel):
    assigned_label: BreezeCallouts | None


CorrodeCallouts = Literal["A Site", "B Site"]
class CorrodeCalloutResult(BaseModel):
    assigned_label: CorrodeCallouts | None


FractureCallouts = Literal["A Site", "B Site"]
class FractureCalloutResult(BaseModel):
    assigned_label: FractureCallouts | None


HavenCallouts = Literal["A Site", "B Site"]
class HavenCalloutResult(BaseModel):
    assigned_label: HavenCallouts | None


IceboxCallouts = Literal["A Site", "B Site"]
class IceboxCalloutResult(BaseModel):
    assigned_label: IceboxCallouts | None


LotusCallouts = Literal["A Site", "B Site"]
class LotusCalloutResult(BaseModel):
    assigned_label: LotusCallouts | None


PearlCallouts = Literal["A Site", "B Site"]
class PearlCalloutResult(BaseModel):
    assigned_label: PearlCallouts | None


SplitCallouts = Literal["A Site", "B Site"]
class SplitCalloutResult(BaseModel):
    assigned_label: SplitCallouts | None


SunsetCallouts = Literal["A Site", "B Site"]
class SunsetCalloutResult(BaseModel):
    assigned_label: SunsetCallouts | None


map_models: dict[str, type[BaseModel]] = {
    "Abyss": AbyssCalloutResult,
    "Ascent": AscentCalloutResult,
    "Bind": BindCalloutResult,
    "Breeze": BreezeCalloutResult,
    "Corrode": CorrodeCalloutResult,
    "Fracture": FractureCalloutResult,
    "Haven": HavenCalloutResult,
    "Icebox": IceboxCalloutResult,
    "Lotus": LotusCalloutResult,
    "Pearl": PearlCalloutResult,
    "Split": SplitCalloutResult,
    "Sunset": SunsetCalloutResult,
}
