from sqlalchemy import Column, Integer, DateTime, Float
from sqlalchemy.orm import sessionmaker
from utils import engine, Base


Session = sessionmaker(bind=engine)

session = Session()


class Record(Base):
    __tablename__ = 'record'

    id = Column(Integer, primary_key=True)

    date_time = Column(DateTime, unique=True)

    threshold = Column(Float)

    eccentricity = Column(Float)

    size = Column(Float)

    mean_intensity = Column(Float)

    min_intensity = Column(Float)

    section_one_eccentricity = Column(Float)

    section_one_size = Column(Float)

    section_one_mean_intensity = Column(Float)

    section_one_min_intensity = Column(Float)

    section_two_eccentricity = Column(Float)

    section_two_size = Column(Float)

    section_two_mean_intensity = Column(Float)

    section_two_min_intensity = Column(Float)

    section_three_eccentricity = Column(Float)

    section_three_size = Column(Float)

    section_three_mean_intensity = Column(Float)

    section_three_min_intensity = Column(Float)

    def save(self):

        session.add(self)

        session.commit()

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
