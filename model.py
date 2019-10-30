import datetime
from sqlalchemy import Column, Integer, DateTime, \
    Float, ForeignKey, String
from sqlalchemy.orm import relationship
from utils import Base
from utils import session


class Record(Base):
    __tablename__ = 'record'

    id = Column(Integer, primary_key=True)

    date_time = Column(DateTime, unique=True)

    poredata = relationship("PoreData", back_populates="record")

    mean_eccentricity = Column(Float)

    std_eccentricity = Column(Float)

    mean_size = Column(Float)

    std_size = Column(Float)

    mean_mean_intensity = Column(Float)

    std_mean_intensity = Column(Float)

    mean_min_intensity = Column(Float)

    std_min_intensity = Column(Float)

    mean_major_axis_length = Column(Float)

    std_major_axis_length = Column(Float)

    mean_minor_axis_length = Column(Float)

    std_minor_axis_length = Column(Float)

    mean_orientation = Column(Float)

    std_orientation = Column(Float)

    mean_weighted_centroid = Column(String)

    std_weighted_centroid = Column(String)

    created_date = Column(DateTime, default=datetime.datetime.utcnow)

    def save(self):

        if not self.id:
            record_query = session.query(Record)

            record_query = record_query.filter(
                Record.date_time == self.date_time
            )

            kk = record_query.one_or_none()

            if kk:
                return kk

        session.add(self)

        session.commit()

        return self

    @staticmethod
    def find_by_date(date_object):

        record_query = session.query(Record)\
            .filter(Record.date_time == date_object)

        return record_query.one_or_none()

    @staticmethod
    def get_all(date_object_lower=None, date_object_upper=None):

        record_query = session.query(Record)

        if date_object_lower:
            record_query = record_query.filter(
                Record.date_time >= date_object_lower
            )

        if date_object_upper:
            record_query = record_query.filter(
                Record.date_time < date_object_upper
            )

        return record_query.order_by(Record.date_time).all()


class PoreData(Base):
    __tablename__ = 'poredata'

    id = Column(Integer, primary_key=True)

    record_id = Column(Integer, ForeignKey('record.id'))

    record = relationship("Record", back_populates="poredata")

    poresectiondata = relationship(
        "PoreSectionData", back_populates="poredata"
    )

    k = Column(Float)

    threshold = Column(Float)

    eccentricity = Column(Float)

    size = Column(Float)

    mean_intensity = Column(Float)

    min_intensity = Column(Float)

    major_axis_length = Column(Float)

    minor_axis_length = Column(Float)

    inertia_tensor_eigvals = Column(String)

    orientation = Column(Float)

    weighted_centroid = Column(String)

    centroid = Column(String)

    created_date = Column(DateTime, default=datetime.datetime.utcnow)

    def save(self):

        session.add(self)

        session.commit()

    @staticmethod
    def find_by_k(k):

        record_query = session.query(Record)\
            .filter(Record.k == k)

        return record_query.one_or_none()


class PoreSectionData(Base):
    __tablename__ = 'poresectiondata'

    id = Column(Integer, primary_key=True)

    pore_data_id = Column(Integer, ForeignKey('poredata.id'))

    poredata = relationship("PoreData", back_populates="poresectiondata")

    k = Column(Float)

    threshold = Column(Float)

    eccentricity = Column(Float)

    size = Column(Float)

    mean_intensity = Column(Float)

    min_intensity = Column(Float)

    created_date = Column(DateTime, default=datetime.datetime.utcnow)

    def save(self):

        session.add(self)

        session.commit()

    @staticmethod
    def find_by_k_and_id(k):

        record_query = session.query(Record)\
            .filter(Record.k == k).filter(Record.id == id)

        return record_query.one_or_none()
